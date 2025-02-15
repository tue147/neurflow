import numpy as np
import torch
import torch.nn as nn

from matplotlib import pyplot as plt

from captum.attr import IntegratedGradients
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score

import math
from typing import Dict, List, Set, Optional, Tuple, Callable, Union

from utils import batch_inference, show, create_aug_dataset

from tqdm import tqdm

metric = "euclidean"
linkage = "ward"

def log_barrier(x: np.ndarray, max_x: np.ndarray, epsilon: float = 1e-6) -> float:
    """
    Calculate the mean of the logarithm of the ratio of the inputs to the absolute value of the maximum input plus epsilon.

    Args:
        x (np.ndarray): The input array.
        max_x (np.ndarray): The maximum value of the input array.
        epsilon (float, optional): The small value added to prevent division by zero. Defaults to 1e-6.

    Returns:
        float: The mean of the logarithm of the ratio of the inputs to the absolute value of the maximum input plus epsilon.
    """
    return np.mean(np.log(np.maximum(x, 0) / (np.abs(max_x) + epsilon) + epsilon)).item()

def accuracy(tensor1: np.ndarray, tensor2: np.ndarray):
    """
    Calculates the accuracy of two tensors.

    Args:
        tensor1 (np.ndarray): The ground truth tensor.
        tensor2 (np.ndarray): The output tensor.

    Returns:
        float: The accuracy as a decimal value between 0 and 1.
    """
    if tensor1.shape[0] == 0:
        return 0
    common_elements = np.intersect1d(tensor1, tensor2).shape[0]
    return common_elements / tensor1.shape[0]

def _wrapper(tensor: torch.Tensor) -> torch.Tensor:
    """
    Wraps the input tensor.

    Args:
        tensor (torch.Tensor): The input tensor with 2, 3 or 4 dimensions.

    Returns:
        torch.Tensor: The wrapped 2D tensor.
    """
    assert len(tensor.shape) > 1 and len(tensor.shape) < 5 # (2, 3, 4)
    if len(tensor.shape) == 3: # transformer to 2D tensor
        tensor = torch.mean(tensor, dim=2)
    elif len(tensor.shape) == 4:
        tensor = torch.mean(tensor, dim=(2,3))
    return tensor


class Model_wrapper(nn.Module):
    '''
    This class is used to wrap the model so that the output of the model is always 2D.
    (We consider each neuron as a node, then we use Integrated Gradients to compute the attribution map for each node).
    '''
    def __init__(self, model: nn.Module) -> None:
        super(Model_wrapper, self).__init__()
        self.model = model
    def forward(self, x):
        out = self.model(x)
        return _wrapper(out)
    
class CriticalNeuron():
    def __init__(self, layer_index: int, neuron_id: int) -> None:
        """
        Initialize a Critical_neu object.

        Args:
            layer_index (int): The index of the layer the critical neuron belongs to.
            neuron_id (int): The ID of the critical neuron.
        """
        self.layer_index = layer_index
        self.neuron_id = neuron_id
        self.top_image_indices: Dict[int, List[int]] = {}  # dict(semantic label : indices of top activated images)
        self.lower_layer_cneus: Dict[int, Dict[int, float]] = {}  # dict(semantic label : dict(neurons at the lower layer: score))
        self.activation_values: Dict[int, float] = {}  # dict(semantic label : mean activation value)
        self.concept_vectors: Dict[int, torch.Tensor] = {}  # dict(semantic label : concept vector)
        self.label_scores: Dict[int, float] = {}  # dict(semantic label : score)

    def get_list_labels(self):
        return list(self.top_image_indices.keys())
    
    def get_top_imgs(self, label = -1):
        if label == -1:
            return self.top_image_indices
        return self.top_image_indices[label]
    
    def set_top_imgs(self, label: int, top_imgs: list[int] | int):
        if isinstance(top_imgs, list):
            self.top_image_indices[label] = top_imgs
        else:
            if len(self.top_image_indices[label]) == 0:
                self.top_image_indices[label] = [top_imgs]
            else:
                self.top_image_indices[label].append(top_imgs)
            
    def get_lower_layer_cneus(self, label = -1):
        if label == -1:
            return self.lower_layer_cneus
        return self.lower_layer_cneus[label].keys()
    
    def get_score(self, label, lower_layer_cneu):
        return self.lower_layer_cneus[label][lower_layer_cneu]
    
    def get_activation_value(self, label = -1):
        if label == -1:
            return self.activation_values
        return self.activation_values[label]
    
    def get_concept_vector(self, label = -1):
        if label == -1:
            return self.concept_vectors
        return self.concept_vectors[label]

class Framework():
    def __init__(
        self,
        start_layer_index: int,
        end_layer_index: int,
        data: torch.Tensor,  
        class_id: int,
        device: str = "cpu",
        granularities: List[float] = [0.25, 0.5, 1],
        stride_ratio: float = 0.5,
        augment: bool = False
    ) -> None:
        """
        Initialize the framework.

        Args:
            start_layer_index (int): The index of the start layer.
            end_layer_index (int): The index of the end layer.
            data (torch.Tensor): The input data.
            device (str, optional): The device. Defaults to "cpu".
            granularities (List[float], optional): The granularities. Defaults to [0.3, 0.5, 0.8, 1].
            stride_ratio (float, optional): The stride ratio. Defaults to 0.5.
            augment (bool, optional): Whether to augment the data. Defaults to False.
        """
        assert len(data.shape) == 4
        self.img_size = data.shape[-1]
        self.start_layer = start_layer_index
        self.end_layer = end_layer_index
        self.data = data
        self.class_id = class_id
        self.device = device
        self.granularities = granularities
        self.stride_ratio = stride_ratio
        self.augment = augment
        self.layers = {
            layer: {} for layer in range(start_layer_index, end_layer_index + 1)
        } # Dict[layer, Dict[neuron_id, CriticalNeuron]]
        if augment:
            self.concept_data = create_aug_dataset(self.get_crop_data())
        else:
            self.concept_data = self.get_crop_data()

    def fit(
        self,
        current_layer: int,
        netA: torch.nn.Module,
        netB: torch.nn.Module,
        top_images_per_neuron: int = 50,
        max_top_neurons: int = -1,
        acceptable_error_range = 0.05,
        batch_size_ig: int = 128,
        batch_size: int = 128,
    ) -> None:
        """
        Fit the framework for the given layer.

        Args:
            current_layer (int): The current layer index.
            netA (nn.Module): The model representing all of the previous layers.
            netB (nn.Module): The model representing the intermediate layer.
            top_images_per_neuron (int, optional): The number of top images per neuron. Defaults to 50.
            max_top_neurons (int, optional): The maximum number of critical neurons. Defaults to -1: equals to no constraint.
            acceptable_error_range (float, optional): The error range for prunning. Default to 0.05.
            batch_size_ig (int, optional): The batch size for calculate the IntegratedGradient scores. Default to 128.
            batch_size (int, optional): The batch size for calculate the activaton. Default to 128. 
        """
        if self.is_fitted(layer=current_layer):
            return

        if current_layer == self.end_layer:
            self.layers[self.end_layer][self.class_id] = CriticalNeuron(self.end_layer, self.class_id)

        self._find_critical_neurons(
            current_layer, self.concept_data, netA, netB, top_images_per_neuron, 
            max_top_neurons, acceptable_error_range, batch_size_ig, batch_size,
        )

    def _get_integrated_gradients(
        self,
        syn_data: torch.Tensor, 
        neu_id: int,
        netB: torch.nn.Module, 
        batch_size: int = 128  
    ) -> torch.Tensor: 
        '''
        Return the importance of each lower layer neuron for each upper layer neuron.

        Args:
            syn_data (torch.Tensor): The input data to the intermediate layer model.
            neu_id (int): The index of the upper layer neuron.
            netB (torch.nn.Module): The model representing the intermediate layer.
            batch_size (int): The size of each batch for processing.

        Returns:
            torch.Tensor: The importance of each lower layer neuron for the upper layer neuron.
        '''
        integrated_gradients = IntegratedGradients(netB.to(self.device))
        importance = []
        for i in range(0, len(syn_data), batch_size):
            batch_data = syn_data[i:i + batch_size].to(self.device)
            batch_data.requires_grad = True
            attributions_ig = integrated_gradients.attribute(
                batch_data, baselines=batch_data * 0, target=neu_id
            )
            importance.append(
                _wrapper(attributions_ig.detach().cpu())
            )
            batch_data.cpu().detach()
        return torch.cat(importance, dim=0)
    
    def _find_critical_neurons(
        self,
        layer: int, 
        concept_data: torch.Tensor,
        netA: nn.Module,
        netB: nn.Module,
        num_top_imgs: int = 50,
        max_top_neurons: int = -1,
        acceptable_error_range: float = 0.05, 
        batch_size_ig: int = 128,
        batch_size: int = 128,
        max_n_clusters: int = 5,
        num_top_img_show: int = 5,
    ) -> None:
        '''
        Find the set of critical neurons at the lower layer.
        Find the concept vectors for each neuron at the current layer.

        Args:
            layer (int): The current layer.
            concept_data (torch.Tensor): The input data to the current layer.
            netA (nn.Module): The model representing all of the previous layers.
            netB (nn.Module): The model representing the intermediate layer.
            num_top_imgs (int, optional): The number of top images to consider for each neuron. Defaults to 50.
            max_top_neurons (int, optional): The maximum number of critical neurons. Default to -1: equals to no constraint.
            acceptable_error_range (float, optional): The error range for prunning. Default to 0.05.
            batch_size_ig (int, optional): The batch size for calculate the IntegratedGradient scores. Default to 128.
            batch_size (int, optional): The batch size for calculate the activaton. Default to 128. 
            max_n_clusters (int, optional): The maximum number of clusters considered. Defaults to 5.
            num_top_img_show (int, optional): The number of images to show for each neuron. Defaults to 5.

        Returns:
            None.
        '''
        if not self.layers[layer]:
            return
        netB_wrapped = Model_wrapper(netB).eval()
        netA = netA.to(self.device).eval()
        concept_data.requires_grad = False

        prev_layer_activation = batch_inference(netA, concept_data, device=self.device, batch_size=batch_size)
        activations = torch.transpose(batch_inference(netB_wrapped, prev_layer_activation, device=self.device, batch_size=batch_size), 0, 1)
        prev_layer_activation = prev_layer_activation.detach().cpu()
        max_top_neurons = prev_layer_activation.shape[1] if max_top_neurons == -1 else max_top_neurons
        max_top_neurons = min(max_top_neurons, prev_layer_activation.shape[1])

        set_lower_layer_critical_neus = set()
        for neu, critical_neu in tqdm(self.layers[layer].items()):
            top_imgs = torch.topk(activations[neu], num_top_imgs)[1].detach().cpu()
            
            list_scores = self._get_integrated_gradients(
                prev_layer_activation[top_imgs], neu, netB_wrapped, batch_size=batch_size_ig
            )
            _, indices = torch.topk(torch.mean(torch.abs(list_scores), dim=0), max_top_neurons)
            cneu_indices_top = [index.item() for index in indices]
    
            indices_top = self._find_optimal_number_of_critical_neurons(
                neu, netB_wrapped, prev_layer_activation, top_imgs, num_top_imgs, 
                cneu_indices_top, acceptable_error_range, batch_size # type: ignore
            )
            
            set_lower_layer_critical_neus.update(indices_top)
            intermediate_vecs = _wrapper(prev_layer_activation[top_imgs]).numpy()

            best_n_clusters = self._find_optimal_number_of_clusters(intermediate_vecs, max_n_clusters)
            
            base_cmodel = AgglomerativeClustering(
                n_clusters=best_n_clusters,metric=metric,linkage=linkage # type: ignore
            ).fit(intermediate_vecs)
            base_labels = base_cmodel.labels_
                
            self._store_critical_neuron_info(
                critical_neu, 
                base_labels, 
                activations, 
                intermediate_vecs, 
                list_scores, 
                top_imgs.numpy(), 
                indices_top, 
                num_top_img_show,
            )
            
        if layer != self.start_layer:
            for lower_layer_cneus in set_lower_layer_critical_neus:
                self.layers[layer-1][lower_layer_cneus] = CriticalNeuron(layer-1, lower_layer_cneus)
    
    def _find_optimal_number_of_critical_neurons(
        self,
        neu: int,
        netB: nn.Module,
        prev_layer_activation: torch.Tensor,
        top_imgs: torch.Tensor, # type: ignore
        num_top_imgs: int,
        cneu_indices_top: List[int],
        acceptable_error_range: float = 0.05,
        batch_size: int = 128,
    ) -> List[int]:
        """
        Find the optimal number of top neurons.

        Args:
            neu (int): The target neuron.
            netB (nn.Module): The intermediate layer (wrapped by Wrapper class).
            prev_layer_activation (torch.Tensor): The activation of the previous layer.
            top_imgs (torch.Tensor): The indices of top images.
            num_top_imgs (int): The number of top images.
            cneu_indices_top (List[int]): The indices of the top neurons.
            acceptable_error_range (float, optional): The error range for prunning. Defaults to 0.05.
            batch_size (int, optional): The batch size. Default to 128.

        Returns:
            List[int]: The indices of the top neurons.
        """
        optimal_num_top_neu = 0
        top_imgs: np.ndarray = top_imgs.numpy()
        accs = []
        if acceptable_error_range >= 0:
            mask = torch.ones_like(prev_layer_activation[0])
            for index in cneu_indices_top:
                mask[index] = 0
                out = batch_inference(netB, prev_layer_activation * mask, device = self.device, batch_size=batch_size).detach().cpu()
                top_imgs_after_masked = torch.topk(out[:, neu], num_top_imgs)[1].numpy()
                accs.append(accuracy(top_imgs, top_imgs_after_masked))
            acceptable_acc = min(accs) + acceptable_error_range
            for i in range(len(accs)):
                if accs[i] < acceptable_acc:
                    optimal_num_top_neu = i+1
                    break
        indices_top = cneu_indices_top[:optimal_num_top_neu if optimal_num_top_neu != 0 else len(cneu_indices_top)]
        return indices_top
    
    def _store_critical_neuron_info(
        self,
        critical_neu: CriticalNeuron,
        labels: np.ndarray,
        activation: torch.Tensor,
        intermediate_vecs: np.ndarray,
        scores: torch.Tensor,
        top_imgs: np.ndarray,
        indices_top: List[int],
        num_top_img_show: int,
    ) -> None:
        """
        Store the information of critical neurons.

        Args:
            critical_neu (CriticalNeuron): The critical neuron.
            labels (np.ndarray): The labels of the neurons.
            activation (torch.Tensor): The activation values of the neurons.
            intermediate_vecs (np.ndarray): The intermediate vectors at the previous layer.
            scores (List[int]): The scores of the neurons.
            top_imgs (np.ndarray): The indices of the top images.
            indices_top (List[int]): The indices of the top neurons.
            num_top_img_show (int): The number of top images to show.

        Returns:
            None.
        """
        unique_labels = np.unique(labels)
        critical_neu.lower_layer_cneus = {
            label: {
                index: torch.mean(scores[labels == label], dim=0)[index].item()
                for index in indices_top
            }
            for label in unique_labels
        }
        self._rescale_scores(critical_neu)
        
        for label in unique_labels:
            critical_neu.set_top_imgs(int(label), [])
            
        closest_indices = [
            self._get_closest_indices(
                labels, label, unique_labels, intermediate_vecs[:, indices_top], num_top_img_show
            )
            for label in unique_labels
        ]
        for i, (index, label) in enumerate(zip(top_imgs, labels)):
            if np.isin(i, closest_indices[label]):
                critical_neu.set_top_imgs(int(label), int(index))
                if critical_neu.label_scores.get(self.labels_concept_data[index]) == None:
                    critical_neu.label_scores[self.labels_concept_data[index]] = 0
                critical_neu.label_scores[self.labels_concept_data[index]] += 1
        for label in unique_labels:
            critical_neu.concept_vectors[int(label)] = torch.mean(
                torch.from_numpy(intermediate_vecs[labels == label]), dim=0
            )
            critical_neu.activation_values[int(label)] = torch.mean(
                activation[critical_neu.neuron_id, top_imgs][labels == label],
                dim=0,
            ).detach().cpu().item()

    def _rescale_scores(self, critical_neu: CriticalNeuron) -> None:
        '''
        Rescale the scores of the lower layer critical neurons.
        '''
        for label_dict in critical_neu.lower_layer_cneus.values():
            for index, score in label_dict.items():
                label_dict[index] = score / (sum(abs(other_score) for other_score in label_dict.values())+1e-7)

    def _get_closest_indices(
        self,
        labels: np.ndarray,
        label: int,
        unique_labels: np.ndarray,
        intermediate_vecs: np.ndarray,
        num_top_img_show: int | None = None,
    ) -> np.ndarray:
        """
        Get the indices of the closest images to the center of the cluster.

        Args:
            labels (np.ndarray): The labels of the clustering.
            label (int): The label of the clustering.
            unique_labels (np.ndarray): The unique labels of the clustering.
            intermediate_vecs (np.ndarray): The intermediate vectors at the previous layer.
            num_top_img_show (int, optional): The number of top images to show. Default to None: show the all of the images

        Returns:
            np.ndarray: The indices of the closest images to the center of the cluster.
        """
        cluster_data = intermediate_vecs[labels == label]
        cluster_centers = np.array(
            [intermediate_vecs[labels == label].mean(axis=0) for label in unique_labels]
        )
        distances = cdist(cluster_data, [cluster_centers[label]], metric="euclidean")
        if num_top_img_show != None:
            closest_idx = np.argsort(distances.flatten())[:num_top_img_show]
        else:
            closest_idx = np.argsort(distances.flatten())
        original_indices = np.where(labels == label)[0]
        return original_indices[closest_idx]
    
    def cluster_concept_vectors(
        self,
        layer: int,
        n_clusters: Optional[int] = None,
        num_top_vecs: Optional[int] = None,
        max_n_clusters: int = 100,
    ) -> Tuple[Dict[int, Dict[int, int]], int]:
        """
        Cluster the concept vectors of the critical neurons at the current layer.

        Args:
            layer (int): The current layer.
            n_clusters (int, optional): The number of clusters. Defaults to None.
            num_top_vecs(int, optional): The number of top vectors for each cluster. Defaults to None: yield all of the vecs.
            max_n_clusters (int, optional): The maximum number of clusters. Defaults to 100.

        Returns:
            tuple:
                - Dict[int, Dict[int, int]]: {neuron_id: {semantic_label: node_combination_label}}
                - int: The optimal number of clusters when `n_clusters` is None; otherwise, returns the provided `n_clusters` value.
        """
        if not self.is_fitted(layer):
            raise ValueError("Please fit layer {} first!".format(layer))

        concept_vectors = []
        neurons =  []
        for neu, critical_neu in self.layers[layer].items():
            for label, vec in critical_neu.concept_vectors.items():
                concept_vectors.append(vec)
                neurons.append((neu, label))
        concept_vectors = torch.stack(concept_vectors, dim=0).numpy()

        if n_clusters is None:
            n_clusters = self._find_optimal_number_of_clusters(concept_vectors, max_n_clusters)
            print("Best number of cluster:", n_clusters)

        combination_neurons_clustering = AgglomerativeClustering(
            n_clusters=n_clusters, metric=metric, linkage=linkage # type: ignore
        ).fit(concept_vectors)
        comb_neu_labels = combination_neurons_clustering.labels_

        clusters = [
            self._get_closest_indices(
                comb_neu_labels, label, np.unique(comb_neu_labels), concept_vectors, num_top_vecs
            )
            for label in range(n_clusters)
        ]

        clustered_concepts = {
            neu : {} 
            for neu in self.layers[layer].keys()
        }
        for i, (neu, label) in enumerate(neurons):
            if np.isin(i, clusters[comb_neu_labels[i]]):
                clustered_concepts[neu][label] = int(comb_neu_labels[i])
        return clustered_concepts, n_clusters

    def get_concept_vector(self, layer, neuron_id, cluster_label):
        """
        Get the concept vector corresponding to the cluster label of a neuron.

        Args:
            layer (int): The layer index.
            neuron_id (int): The neuron ID.
            cluster_label (int): The cluster label.

        Returns:
            np.ndarray: The concept vector.

        Raises:
            ValueError: If the layer is not fitted, the neuron is not found,
                or the cluster label is not found.
        """
        if self.is_fitted(layer):
            neuron = self.layers[layer].get(neuron_id)
            if neuron and cluster_label in neuron.concept_vectors:
                return neuron.concept_vectors[cluster_label]
        raise ValueError(f"Layer {layer} or neuron {neuron_id} not fitted, "
                         f"or cluster label {cluster_label} not found.")

    def debug_img(
        self, 
        image: torch.Tensor, 
        layer: int, 
        net_current_layer: nn.Module, 
        n_cluster: Optional[int] = None, 
        threshold: float = 0.8, 
        num_top_vecs: Optional[int] = 5,
        specific_concept: Optional[int] = None,
        granularities: List[float] = [0.25, 0.5, 1], 
        stride_ratio: float = 0.50, 
        num_show_per_concept: int = 1, 
        num_imgs_show: int = 5,
        max_n_clusters: int = 100, 
        batch_size: int = 128,
    ) -> None:
        '''
        Debug the input image: find the most relevant concept vectors at a layer.

        Args:
            image (torch.Tensor): The debugging image.
            layer (int): The layer index.
            net_current_layer (nn.Module): The sub-model from the start to the current layer.
            n_cluster (int|str, optional): Number of cluster (greater than 2). Default to None: auto find the number of cluster.
            threshold (float, optional): The log barrier threshold for visualization. Default to 0.8.
            num_top_vecs (int, optional): The number of top vectors use in log barier show for each cluster. Defaults to 5.
            specific_concept (int, optional): The specific concept to debug. When specified, the n_cluster must not be None. Default to None. 
            granularities (List[float], optional): The proportion to crop the debugging image. Default to [0.25, 0.5, 1].
            stride_ratio (float, optional): The stride ratio to crop the debugging image. Default to 0.50.
            num_show_per_concept (int, optional): Number of vecs to consider in log barier. Default to 1. 
            num_imgs_show (int, optional): Number of images show for each concept. Default to 5.
            max_n_clusters (int, optional): Maximum number of clusters. Default to 100. 
            batch_size (int, optional): The batch size. Default to 128.

        Returns:
            None.
        '''
        if not self.is_fitted(layer):
            print(f"Please fit layer {layer} first!")
            return
        if specific_concept is not None:
            assert n_cluster is not None
            assert n_cluster > specific_concept, "The specific_concept must be in the range of (0, {})".format(n_cluster)
            
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        img_crops = [
            self._cropping(image, int(granularity * self.img_size), stride_ratio) 
            for granularity in granularities
        ]
        img_crops = torch.cat(img_crops)
        net = Model_wrapper(net_current_layer).eval()
        activations = batch_inference(net, img_crops, device=self.device, batch_size=batch_size).detach().cpu().numpy()
        # the mean concept vectors (for each cluster)
        dict_cluster_concepts, _ = self.cluster_concept_vectors(layer, n_cluster, num_top_vecs, max_n_clusters)
        mean_activation_value = {}
        dict_cluster_label_neu = {}
        list_cneus_current_layer = [neu for neu in self.layers[layer].keys()]

        for cneu in list_cneus_current_layer:
            for label, cluster_label in dict_cluster_concepts[cneu].items():
                if cluster_label not in mean_activation_value:
                    mean_activation_value[cluster_label] = {neu : [] for neu in range(max(list_cneus_current_layer)+1)}
                    dict_cluster_label_neu[cluster_label] = []
                mean_activation_value[cluster_label][cneu].append(self.layers[layer][cneu].get_activation_value(label))
                dict_cluster_label_neu[cluster_label].append(cneu)

        for cluster_label in mean_activation_value.keys():
            mean_activation_value[cluster_label] = np.array(
                list(sum(mean_activation_value[cluster_label][neu])/len(mean_activation_value[cluster_label][neu]) 
                if len(mean_activation_value[cluster_label][neu])!=0 else 0 
                for neu in range(max(list_cneus_current_layer)+1))
            )[dict_cluster_label_neu[cluster_label]]

        # find the most relevant concept vectors
        max_barier_scores = {}
        dict_cluster_for_visualize, _ = self.cluster_concept_vectors(layer, n_cluster, 1, max_n_clusters)
        for cluster_label, mean_activation_vector in mean_activation_value.items():
            if specific_concept is not None and specific_concept != cluster_label:
                continue
            barier_scores = [
                log_barrier(activations[index,dict_cluster_label_neu[cluster_label]], mean_activation_vector) 
                for index in range(activations.shape[0])
            ]
            max_barier_scores[cluster_label] = math.exp(max(barier_scores))
            indices = [score >= math.log(threshold) for score in barier_scores]
            # show the crops
            if np.sum(indices) > 0:
                print("====================")
                print("Cluster label:", cluster_label)
                print("====================")
                show_img = img_crops[indices].detach().cpu()
                for i, img in enumerate(show_img):
                    if i >= num_imgs_show:
                        break
                    plt.subplot(1, min(show_img.shape[0], num_imgs_show), i+1)
                    show(img)
                plt.show()
                # show the concepts
                self.visualize_cluster_concept_vectors(layer, cluster_label, dict_cluster_for_visualize, num_show_per_concept)
        print("Score of each cluster:", max_barier_scores)

    def visualize_cluster_concept_vectors(
        self,
        layer: int,
        cluster_label: int,
        dict_cluster_concepts: Dict[int, Dict[int, int]],
        num_concepts_per_cluster: int = 3,
        num_top_images: int = 5,
    ) -> Dict[int, int]:
        '''
        Show the specified cluster of the concept vectors

        Args:
            layer (int): The layer of the network.
            cluster_label (int): The label of the cluster to visualize.
            dict_cluster_concepts (Dict[int, Dict[int, int]]): A dictionary mapping each neuron to a dictionary of concept labels and their corresponding cluster labels.
            num_concepts_per_cluster (int, optional): The number of concepts to visualize per cluster. Defaults to 3.
            num_top_images (int, optional): The number of top images to visualize. Defaults to 5.
        
        Returns:
            Dict[int, int]: A dictionary mapping each neuron to the semantic label being visualized.
        '''
        if not self.is_fitted(layer):
            raise ValueError(f"Please fit layer {layer} first!")

        count = 0
        visualizing_neu_ids = {}
        for neuron, temp_dict in dict_cluster_concepts.items():
            for semantic_label, node_combination_label in temp_dict.items():
                if node_combination_label == cluster_label:
                    count += 1
                    if count > num_concepts_per_cluster:
                        break
                    print("Feature map:", neuron)
                    self.visualize_top_images(
                        layer, neuron, semantic_label, num_top_images
                    )
                    visualizing_neu_ids[neuron] = semantic_label
            if count >= num_concepts_per_cluster:
                break
        return visualizing_neu_ids

    def _find_optimal_number_of_clusters(
        self,
        vectors: np.ndarray,
        max_clusters: int = 10
    ) -> int:
        '''
        Find the optimal number of clusters via silhouette scores

        Args:
            vectors (np.ndarray): The input vectors.
            max_clusters (int): The maximum number of clusters.

        Returns:
            int: The optimal number of clusters.
        '''
        max_clusters = min(vectors.shape[0] - 1, max_clusters)
        if max_clusters <= 2:
            return 2
        cluster_range = range(2, max_clusters + 1)
        scores = []
        best_score = -1
        best_n_clusters = None

        for n_clusters in cluster_range:
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric="euclidean",
                linkage="ward"
            ).fit(vectors)
            score = silhouette_score(vectors, clustering.labels_)
            scores.append(score)
            if score > best_score:
                best_score = score
                best_n_clusters = n_clusters

        return best_n_clusters # type: ignore
    
    def _crop_data(self, size: int, stride_ratio: float = 0.50) -> torch.Tensor:
        if size == self.img_size:
            return self.data
        return self._cropping(self.data, size, stride_ratio)
    
    def _cropping(self, data: torch.Tensor, size: int, stride_ratio: float = 0.50) -> torch.Tensor:
        '''
        Cropping the input data. 
        '''
        strides = max(min(int(size * stride_ratio), self.img_size - size), 1)
        crops = torch.nn.functional.unfold(data, kernel_size=size, stride=strides)
        crops = crops.transpose(1, 2).contiguous().view(-1, 3, size, size)
        crops = torch.nn.functional.interpolate(crops, size=self.img_size, mode='bilinear', align_corners=False)
        return crops

    def get_crop_data(self) -> torch.Tensor:
        data = []
        labels = []
        for granularity in self.granularities:
            size = int(granularity * self.img_size)
            data.append(self._crop_data(size, self.stride_ratio))
            if self.augment:
                labels.extend([size for _ in range(data[-1].shape[0] * 5)])
            else:
                labels.extend([size for _ in range(data[-1].shape[0])])
        self.labels_concept_data = labels
        return torch.cat(data)
            
    def visualize_top_images(
        self,
        layer: int,
        neuron: int,
        label: int = -1,
        num_top_imgs: int = 5,
    ) -> None:
        """
        Visualize the top images of a neuron.

        Args:
            layer (int): The layer of the neuron.
            neuron (int): The index of the neuron.
            label (int, optional): The label of the top images to visualize. Defaults to -1.
            num_top_imgs (int, optional): The number of top images to visualize. Defaults to 5.

        Returns:
            None.
        """
        if not self.is_fitted(layer):
            print(f"Please fit layer {layer} first!")
            return

        if neuron not in self.layers[layer]:
            print(f"{neuron} is not a critical neuron at layer {layer}!")
            return

        top_images: Dict[int, List[int]] = self.layers[layer][neuron].get_top_imgs()
        labels_to_show: List[Tuple[int, List[int]]] = []

        if label == -1:
            labels_to_show = [(label, indices) for label, indices in top_images.items()]
        elif label in top_images:
            labels_to_show.append((label, top_images[label]))

        for label, indices in labels_to_show:
            print(f"Label: {label}")
            if num_top_imgs > len(indices) or num_top_imgs == -1:
                num_visualize = len(indices)
            else:
                num_visualize = num_top_imgs

            for i, index in enumerate(indices[:num_visualize]):
                plt.subplot(1, num_visualize, i + 1)
                show(self.concept_data[index].detach().cpu().numpy())
            plt.tight_layout()
            plt.show()

    def get_critical_neurons(self, layer: int, neuron_index: int = -1) -> List[int] | Dict[int, Dict[int, float]]:
        '''
        Get the critical neurons in a layer.

        Args:
            layer (int): The layer index.
            neuron_index (int, optional): The neuron index. Default to -1: return all of the critical neurons at the layer.
        
        Returns:
            List[int]: All of the critical neurons.
            or
            Dict[int, Dict[int, float]]: The score of subsequent neurons for each cluster label, has the following structure: Dict[cluster_label, Dict[Lower_layer_neuron, score]].
        '''
        if not self.is_fitted(layer):
            print(f"Please fit layer {layer} first!")
            return []
        critical_neurons = list(self.layers[layer].keys())
        if neuron_index == -1:
            return critical_neurons
        elif neuron_index in self.layers[layer]:
            return self.layers[layer][neuron_index].lower_layer_cneus
        else:
            raise ValueError(f"{neuron_index} is not a critical neuron at layer {layer}")
    
    def get_top_images_for_labels(self, layer: int, neuron: int) -> List[int]:
        '''
        Get the list of top image for each sub concept label.

        Args:
            layer (int): The layer index.
            neuron_index (int): The neuron index.

        Returns:
            List[int]: The indices of the top images.
        '''
        if not self.is_fitted(layer):
            print(f"Please fit layer {layer} first!")
            return []

        if neuron not in self.layers[layer]:
            print(f"{neuron} is not a critical neuron at layer {layer}!")
            return []

        return self.layers[layer][neuron].get_top_imgs()
    
    def is_fitted(self, layer: int) -> bool:
        '''
        Check if a layer fitted.
        '''
        try:
            layer_keys = list(self.layers[layer].keys())
        except:
            return False
        
        return bool(layer_keys) and bool(self.layers[layer][layer_keys[0]].lower_layer_cneus)
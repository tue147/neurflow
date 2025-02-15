import torch
import numpy as np
import torchvision.transforms as transforms
from torch import nn
from PIL import Image
from matplotlib import pyplot as plt
import random
from math import ceil
import warnings
from collections import OrderedDict
from torchvision.datasets import ImageNet
from math import ceil
from typing import Tuple, Dict, List, Any, Union, Callable
from NeurFlow import Framework

def show(img):
    img = np.array(img)
    img -= img.min();img /= img.max()
    img = np.transpose(img, (1, 2, 0))
    plt.imshow(img); plt.axis('off')

def show_PIL(img):
    img = np.array(img)
    img = (img - img.min()) / (img.max() - img.min())
    img *= 255
    img = img.astype(np.uint8)
    plt.imshow(img)
    plt.axis('off')

def tensor_to_pil(tensor):
    """
    Converts a normalized torch tensor to a PIL image.
    
    Args:
        tensor (torch.Tensor): The normalized tensor with shape (C, H, W).
        mean (list or tuple): The mean used for normalization (one value per channel).
        std (list or tuple): The standard deviation used for normalization (one value per channel).

    Returns:
        PIL.Image: The converted PIL image.
    """
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    tensor = tensor.detach().cpu()

    # Undo normalization
    mean = torch.tensor(mean)[:, None, None]
    std = torch.tensor(std)[:, None, None]
    tensor = tensor * std + mean

    # Convert to NumPy array and permute dimensions to (H, W, C)
    img = tensor.permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    
    return img

def pil_to_tensor(pil_image):
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert PIL image to PyTorch tensor
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # Normalize the tensor
    ])
    return transform(Image.fromarray(pil_image)).float() # type: ignore

def randomly_augment_image(image, num_images=5):
    """
    Generate a specified number of randomly augmented images from an input image using OpenCV.

    Parameters:
    - pil_image (PIL Image): Input image to augment.
    - num_images (int): Number of augmented images to generate.
    
    Returns:
    - list of np.array: List containing the augmented images in OpenCV format.
    """
    import cv2
    # image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image = tensor_to_pil(image)
    augmented_images = []
    augmented_images = []
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert PIL image to PyTorch tensor
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # Normalize the tensor
    ])
    
    for _ in range(num_images):
        img = image.copy()
        
        # Random horizontal flipping
        if random.random() > 0.5:
            img = cv2.flip(img, 1)
        
        # Random vertical flipping
        if random.random() > 0.5:
            img = cv2.flip(img, 0)
        
        # Random rotation
        if random.random() > 0.5:
            angle = random.uniform(-60, 60) 
            (h, w) = img.shape[:2]
            (cX, cY) = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        
        # Random translation
        if random.random() > 0.5:
            tx = random.uniform(-0.1, 0.1) * img.shape[1]
            ty = random.uniform(-0.1, 0.1) * img.shape[0]
            M = np.float32([[1, 0, tx], [0, 1, ty]]) # type: ignore
            img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REPLICATE) # type: ignore
        
        # Random scaling
        if random.random() > 0.5:
            scale = random.uniform(0.8, 1.2)
            resized = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            if scale > 1.0:  # If scaling up, crop the center
                startx = resized.shape[1] // 2 - img.shape[1] // 2
                starty = resized.shape[0] // 2 - img.shape[0] // 2
                img = resized[starty:starty+img.shape[0], startx:startx+img.shape[1]]
            else:  # If scaling down, pad the image
                diff_x = img.shape[1] - resized.shape[1]
                diff_y = img.shape[0] - resized.shape[0]
                img = cv2.copyMakeBorder(resized, diff_y//2, diff_y-diff_y//2, diff_x//2, diff_x-diff_x//2, cv2.BORDER_REPLICATE)
        
        augmented_images.append(transform(Image.fromarray(img)).float()) # type: ignore
    
    return torch.stack(augmented_images)

def create_aug_dataset(dataset, num_images=5):
    augmented_dataset = []
    for image in dataset:
        augmented_dataset.append(randomly_augment_image(image, num_images))
    return torch.cat(augmented_dataset)

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

def batch_inference(model, dataset, batch_size=128, device='cpu'):
    nb_batchs = ceil(len(dataset) / batch_size)
    start_ids = [i*batch_size for i in range(nb_batchs)]
    results = []

    model = model.to(device)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with torch.no_grad():
            for i in start_ids:
                x = torch.tensor(dataset[i:i+batch_size])
                x = x.to(device)   
                results.append(model(x).cpu())
                x.cpu()

    results = torch.cat(results)
    return results

def get_data_transforms():
    '''
    Data normalization for the ImageNet dataset.
    '''
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def load_data(
    path: str,
    label_list: list[int],
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Load the data from the ImageNet dataset and filter the data based on the label_list and model prediction.
    '''
    
    transform = get_data_transforms()
    val_set = ImageNet(root=path, split='val', transform=transform)

    images_list = []
    labels_list = []
    count = 0
    if verbose:
        print("Loading data...")
    for images, labels in val_set:
        count += 1
        if count < (min(label_list)*50//1000)*1000:
            continue
        elif count <= ceil((max(label_list)*50+1)/1000)*1000:
            images_list.append(images.unsqueeze(0))
            labels_list.append(torch.tensor([labels]))
        else:
            break

    # Concatenate all the batches
    all_images = torch.cat(images_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)

    return all_images, all_labels
    
    
def load_model_data(
    path: str,
    label_list: list[int],
    model: nn.Module,
    device: str,
    all_images: torch.Tensor | None = None,
    all_labels: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    '''
    Load the images that the model predicted as in the label_list
    '''
    
    if all_images is None and all_labels is None:
        all_images, all_labels = load_data(path, label_list)
    
    with torch.no_grad():
        outputs = batch_inference(model, all_images, device=device)
        _, predicted_labels = torch.max(outputs, 1)

    predicted_labels = predicted_labels.detach().cpu()
    mask = torch.isin(predicted_labels, torch.tensor(label_list))
    
    class_images = all_images[mask] # type: ignore
    class_labels = all_labels[mask] # type: ignore
    
    return class_images, class_labels

def get_crop_data(data, granularities = [0.25, 0.5 ,1]) -> torch.Tensor:
    '''
    Transform data into crops with different granularities.
    '''
    def _cropping(data: torch.Tensor, size: int, stride_ratio: float = 0.50) -> torch.Tensor:
        '''
        Cropping the input data. 
        '''
        strides = max(min(int(size * stride_ratio), 224 - size), 1)
        crops = torch.nn.functional.unfold(data, kernel_size=size, stride=strides)
        crops = crops.transpose(1, 2).contiguous().view(-1, 3, size, size)
        crops = torch.nn.functional.interpolate(crops, size=224, mode='bilinear', align_corners=False)
        return crops
    
    _data = []
    for granularity in granularities:
        size = int(granularity * 224)
        _data.append(_cropping(data, size, 0.5))
        
    return torch.cat(_data, dim=0)

def extract_submodule(parent_module, module_path):
    """
    Extracts a submodule given a hierarchical path string like 'features.8'.
    Returns the submodule and its immediate parent.
    """
    module_names = module_path.split('.')
    submodule = parent_module
    for name in module_names:
        submodule = getattr(submodule, name)
    return submodule

def add_modules(source, first_part, second_part, split_layer, include_split_layer_in_first_part = True, parent_name="", conditional_modules={}):
    """
    Adds a range of layers/modules from `source` to `target`.
    """
    passed_split_layer = False
    for name, module in source.named_children():
        if parent_name == "":
            joined_name = name
        else:
            joined_name =  parent_name + "_" + name
        if joined_name in conditional_modules:
            module = nn.Sequential(OrderedDict([("0", conditional_modules[joined_name]), ("1", module)]))
        if name == split_layer:
            if include_split_layer_in_first_part:
                first_part.add_module(joined_name, module)
            else:
                second_part.add_module(joined_name, module)
            passed_split_layer = True
            continue
        if not passed_split_layer:
            first_part.add_module(joined_name, module)
        else:
            second_part.add_module(joined_name, module)

def get_sub_model(original_model, start_layer_name, end_layer_name, include_split_layer_in_first_part=False, conditional_modules={}):
    """
    Get the sub module of the original_model from start_layer_name to end_layer_name.
    Handles nested modules like `features.8`.
    """
    if include_split_layer_in_first_part and start_layer_name == end_layer_name:
        return nn.Sequential()
    _, second_part = split_model(original_model, start_layer_name, include_split_layer_in_first_part)
    start_layer_split = start_layer_name.split('.')
    end_layer_split = end_layer_name.split('.')
    max_index = min(len(start_layer_split), len(end_layer_split))
    new_end_layer_name = ""
    for index in range(max_index):
        if new_end_layer_name != "":
            new_end_layer_name += "_" + end_layer_split[index]
        else:
            new_end_layer_name = end_layer_split[index]
        if start_layer_split[index] != end_layer_split[index]:
            new_end_layer_name = ".".join([new_end_layer_name] + end_layer_split[index+1:])
            break
    # print(second_part)
    return split_model(second_part, new_end_layer_name, True, conditional_modules)[0]

def _split_model(source, remaining_name, root_name_list, first_part, second_part, include_split_layer_in_first_part=True, conditional_modules={}):
    '''
    Helper function to split the model into two parts.
    '''
    if "." not in remaining_name:
        add_modules(source, first_part, second_part, remaining_name, include_split_layer_in_first_part, 
                    "_".join(root_name_list), conditional_modules)
    else:
        new_root, new_remaining = remaining_name.split('.', 1)
        passed_new_root = False
        for name, module in source.named_children():
            if len(root_name_list) == 0:
                joined_name = name
            else:
                joined_name =  "_".join(root_name_list) + "_" + name
            if name in conditional_modules:
                module = nn.Sequential(OrderedDict([("0", conditional_modules[joined_name]), ("1", module)]))
            if name == new_root:
                root_name_list.append(new_root)
                _split_model(extract_submodule(source, new_root), new_remaining, root_name_list, first_part, 
                             second_part, include_split_layer_in_first_part, conditional_modules)
                root_name_list.pop()
                passed_new_root = True
                continue
            if not passed_new_root:
                first_part.add_module(joined_name, module)
            else:
                second_part.add_module(joined_name, module)

def split_model(
    original_model: nn.Module, 
    split_layer_name: str, 
    include_split_layer_in_first_part: bool = True, 
    conditional_modules: dict = {},
):
    """
    Splits a model into two parts at the specified split layer.
    Handles nested modules like `features.8`.
    
    Args:
    - original_model (nn.Module): The original model.
    - split_layer_name (str): The name of the layer to split the model.
    - include_split_layer_in_first_part (bool): Whether to include the split layer in the first part.
    - conditional_modules (dict): The conditional modules to add before a layer. 
            This has the structure of {layer_name: module_to_add_before_the_"layer_name"}.
    """
    first_part = nn.Sequential(OrderedDict())
    second_part = nn.Sequential(OrderedDict())
    
    _split_model(
        original_model, split_layer_name, [], first_part, second_part, include_split_layer_in_first_part, conditional_modules
    )

    return first_part, second_part

def get_conditional_modules(model_name: str) -> Dict:
    '''
    Helper function to split the model correctly.
    This function gives the modules to add to the model before a layer (that why it is called conditional modules).
    '''
    if model_name == "resnet50":
        conditional_modules = {"fc": nn.Flatten()}
    elif model_name == "googlenet":
        class TransformInput(nn.Module):
            def __init__(self, transform_input=True):
                super(TransformInput, self).__init__()
                self.transform_input = transform_input

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                if self.transform_input:
                    x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
                    x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
                    x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
                    x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
                return x
    
        conditional_modules = {"fc": nn.Flatten(), "conv1": TransformInput()}
    elif model_name == "alexnet":
        conditional_modules = {"classifier_0": nn.Flatten(), "classifier": nn.Flatten()}
    else:
        raise ValueError("Model not supported")
    return conditional_modules

def visualize_chain_of_concepts(
    FW: Framework, 
    concept_id: int, 
    start_layer: int, 
    end_layer: int, 
    num_top_concepts: int, 
    list_num_cluster: List[int] | None = None, 
    specify_cluster: bool = True, 
    num_show_per_concept: int = 1, 
    num_top_vecs: int | None = 10,
):
    '''
    This function visualizes the chain of concepts.
    
    Args:
    - FW (Framework): The framework object.
    - concept_id (int): The concept id.
    - start_layer (int): The start layer.
    - end_layer (int): The end layer.
    - num_top_concepts (int): The number of top concepts.
    - list_num_cluster (List[int], optional): The list of number of clusters. Defaults to None.
    - specify_cluster (bool, optional): Whether to specify the number of clusters. Defaults to True.
    - num_show_per_concept (int, optional): The number of examples to show. Defaults to 1.
    - num_top_vecs (int, optional): The number of representative vectors for each concept. Defaults to 10.
    '''
    if list_num_cluster is None:
        assert specify_cluster==False, "The list number of clusters must be specified if specify_cluster is True"
    if specify_cluster == True:
        assert list_num_cluster is not None, "The number of clusters must be specified if specify_cluster = True"
        assert len(list_num_cluster) == end_layer+1-start_layer, "The length of list_num_cluster must be equal to the number of layers"
        for num_cluster in list_num_cluster:
            assert num_cluster >= 2, "The number of clusters must be greater than or equal to 2"
        assert list_num_cluster[-1] > concept_id, "The concept_id must be in the range of (0, {})".format(list_num_cluster[-1])
    from contextlib import redirect_stdout
    import io
    dict_labels_all_layers = {}
    dict_labels_all_layers_for_visualize = {}
    
    for layer in range(end_layer, start_layer-1, -1):
        with redirect_stdout(io.StringIO()): # suppress unwanted printing
            dict_labels, best_num_cluster = FW.cluster_concept_vectors(
                layer, 
                list_num_cluster[layer-1-end_layer] if specify_cluster else None, # type: ignore
                num_top_vecs=num_top_vecs,
            )
            
            dict_labels_for_visualize, _ = FW.cluster_concept_vectors(
                layer, 
                best_num_cluster, 
                num_top_vecs=1,
            )
            
        dict_labels_all_layers[layer] = dict_labels
        dict_labels_all_layers_for_visualize[layer] = dict_labels_for_visualize
        
        if not specify_cluster:
            print(f"The best number of cluster at layer {layer}:", best_num_cluster)

    list_critical_cluster_labels = [concept_id]
    FW.visualize_cluster_concept_vectors(
        end_layer, 
        list_critical_cluster_labels[0], 
        dict_labels_all_layers_for_visualize[end_layer], 
        num_show_per_concept,
    )
    for layer in range(end_layer, start_layer, -1):
        print(f"=========layer:{layer-1}===========")
 
        dict_labels_cur = dict_labels_all_layers[layer]
        dict_labels_prev = dict_labels_all_layers[layer-1]
        
        scores = _aggregate_scores(FW, layer, dict_labels_cur, dict_labels_prev)   
        list_critical_cluster_labels = _extract_top_concepts(
            scores, list_critical_cluster_labels, num_top_concepts
        )

        for critical_cluster_label in list_critical_cluster_labels:
            print("------Layer:", layer-1, "cluster_label:", critical_cluster_label, "------")
            FW.visualize_cluster_concept_vectors(
                layer-1, 
                critical_cluster_label, 
                dict_labels_all_layers_for_visualize[layer-1], 
                num_show_per_concept,
            )

def get_neu_id_top_concepts(
    FW: Framework, 
    upper_layer: int, 
    upper_num_cluster: int, 
    lower_num_cluster: int, 
    concept_id: int, 
    num_top_concepts: int, 
    specify_cluster: bool = True, 
    num_top_vecs: int | None = None
):
    '''
    Visualize the upper layer and the lower layer of the top concepts.
    And return the neuron ids of the top concepts.
    
    Args:
    - FW (Framework): The framework object.
    - upper_layer (int): The upper layer.
    - upper_num_cluster (int): The number of clusters at the upper layer.
    - lower_num_cluster (int): The number of clusters at the lower layer.
    - concept_id (int): The concept id.
    - num_top_concepts (int): The number of top concepts.
    - specify_cluster (bool, optional): Whether to specify the number of clusters. Defaults to True.
    - num_top_vecs (int, optional): The number of representative vectors for each concept. Defaults to None.
    '''
    scores, dict_labels_cur_for_visualize, dict_labels_prev_for_visualize = _aggregate_scores_consecutive_layers(
        FW, upper_layer, upper_num_cluster, lower_num_cluster, specify_cluster, num_top_vecs
    )
    print("========= Layer:", upper_layer, "Cluster_label:", concept_id, "=========")
    neu_id_cur = FW.visualize_cluster_concept_vectors(
        upper_layer, concept_id, dict_labels_cur_for_visualize, 1
    )
    list_critical_cluster_labels = _extract_top_concepts(scores, [concept_id], num_top_concepts)
    from collections import defaultdict

    neu_id_prev = defaultdict(list)
    for critical_cluster_label in list_critical_cluster_labels:
        print("========= Top concept:", critical_cluster_label, "=========")
        new_data = FW.visualize_cluster_concept_vectors(
            upper_layer-1, critical_cluster_label, dict_labels_prev_for_visualize, 1
        )
        for key, value in new_data.items():
            neu_id_prev[key].append(value)
    return neu_id_cur, neu_id_prev

def _extract_top_concepts(scores, list_critical_cluster_labels, num_top_concepts):
    '''
    Helper function to extract top concepts at the previous layer.
    '''
    temp = []
    for critical_cluster_label in list_critical_cluster_labels:
        top_dict = {
            key:value for key, value in sorted(
                scores[critical_cluster_label].items(), 
                key=lambda item: item[1], reverse=True
            )[:num_top_concepts]
        }
        print("Top concepts of", critical_cluster_label, "are:", top_dict)
        temp.extend(list(top_dict.keys()))
    return list(set(temp))

def _aggregate_scores_consecutive_layers(FW, upper_layer, upper_num_cluster, lower_num_cluster, specify_cluster=True, num_top_vecs=None):
    '''
    Aggregate the scores of the node combination at the previous layer
    '''
    assert upper_layer > 1, "The upper_layer must be greater than 1"
    assert isinstance(upper_num_cluster, int) == isinstance(lower_num_cluster, int), "The upper and lower number of clusters must be the same type"
    if upper_num_cluster is None:
        assert specify_cluster==False, "The list number of clusters must be specified if specify_cluster is True"
    if specify_cluster == True:
        assert isinstance(upper_num_cluster, int) and isinstance(lower_num_cluster, int), "The number of clusters must be specified if specify_cluster = True"
        assert upper_num_cluster > 1 and lower_num_cluster > 1, "The number of clusters must be greater than 1"
    dict_labels_cur, best_n_clusters = FW.cluster_concept_vectors(
        upper_layer, upper_num_cluster if specify_cluster else None, num_top_vecs=num_top_vecs
    )
    dict_labels_cur_for_visualize, _ = FW.cluster_concept_vectors(
        upper_layer, best_n_clusters, num_top_vecs=1
    )
    dict_labels_prev, best_n_clusters = FW.cluster_concept_vectors(
        upper_layer-1, lower_num_cluster if specify_cluster else None, num_top_vecs=num_top_vecs
    )
    dict_labels_prev_for_visualize, _ = FW.cluster_concept_vectors(
        upper_layer-1, best_n_clusters, num_top_vecs=1
    )
    return _aggregate_scores(FW, upper_layer, dict_labels_cur, dict_labels_prev), dict_labels_cur_for_visualize, dict_labels_prev_for_visualize

def _aggregate_scores(FW, layer, dict_labels_cur, dict_labels_prev):
    '''
    Aggregate the scores of the node combination at the previous layer
    '''
    scores = {}
    for cneu_at_layer in dict_labels_cur.keys(): # dict_labels_cur: dict(neurons at the current layer: dict(semantic label : node combination label))
        cneus_at_prev = FW.get_critical_neurons(layer, cneu_at_layer) # dict(semantic label : dict(neurons at the lower layer: score))
        scores_for_each_cneu_at_layer = {}

        for label_neu_cur, label_score_dict in cneus_at_prev.items(): # (semantic label, dict(neurons at the lower layer: score))
            scores_for_each_cneu_at_layer.setdefault(label_neu_cur, {})

            for cneu, score in label_score_dict.items(): # aggregate the score for each cluster label
                for cluster_label_prev in dict_labels_prev[cneu].values(): # dict_labels_prev: dict(neurons at the previous layer: dict(semantic label previous layer : node combination label previous layer))
                    scores_for_each_cneu_at_layer[label_neu_cur].setdefault(cluster_label_prev, 0) # Dict(semantic label: Dict(node combination label previous layer: aggregated score))
                    scores_for_each_cneu_at_layer[label_neu_cur][cluster_label_prev] += score 

        for label_neu_cur, cluster_label_cur in dict_labels_cur[cneu_at_layer].items(): # (semantic label, node combination label)
            scores.setdefault(cluster_label_cur, {})
            dict_score = scores_for_each_cneu_at_layer[label_neu_cur] # dict(node combination label previous layer: aggregated score)

            for cluster_label_prev, score in dict_score.items(): # dict(node combination label previous layer: aggregated score)
                scores[cluster_label_cur].setdefault(cluster_label_prev, []).append(score) # dict(node combination label current layer: dict(node combination label previous layer: list of aggregated scores)

    # sum the scores
    scores = {
        outer_key: {inner_key: sum(inner_value) for inner_key, inner_value in inner_dict.items()}
        for outer_key, inner_dict in scores.items()
    }
    return scores # dict(node combination label current layer: dict(node combination label previous layer: aggregated score))
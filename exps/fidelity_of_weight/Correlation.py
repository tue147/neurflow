# %%
import torch
from utils import split_model, get_sub_model, batch_inference, get_conditional_modules, load_model_data, accuracy, _wrapper, Model_wrapper, get_crop_data
import numpy as np
import argparse
import time

parser = argparse.ArgumentParser(description="Fidelity of edge weights")

parser.add_argument('--method_name', type=str, required=True, choices=["integrated_gradients", "smoothgrad", "guided_backprop", "gradient_shap", "knockoff", "lrp", "saliency"], help='Type of method to use')
parser.add_argument('--data_dir', type=str, required=True, help='Directory to store the data')
parser.add_argument("--model_name", type=str, default="googlenet", help="Model name to use")
parser.add_argument("--label_list", type=int, nargs='+', default=0, help="List of labels (space-separated values)")
parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for processing")
parser.add_argument("--list_tau", type=int, nargs='+', default=[1, 5, 10, 20, 50], help="Tau value")
parser.add_argument("--num_comb", type=int, default=500, help="Number of combinations")
parser.add_argument("--output_dir", type=str, default="./correlation/", help="Directory to save output")
parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
parser.parse_args()

args = parser.parse_args()

method_name = args.method_name
data_dir = args.data_dir
model_name = args.model_name
label_list = args.label_list
batch_size = args.batch_size
list_lengths = args.list_tau
N = args.num_comb
output_dir = args.output_dir

device = args.device if torch.cuda.is_available() else "cpu"

from torchvision import models
if model_name == "resnet50":
    model = models.resnet50(pretrained=True).eval()
    all_layers = ["layer4.2", "layer4.1", "layer4.0", "layer3.5", "layer3.4", 
                  "layer3.3", "layer3.2", "layer3.1", "layer3.0", "layer2.3", 
                  "layer2.2", "layer2.1", "layer2.0", "layer1.2", "layer1.1", 
                  "layer1.0", "conv1"]
    conditional_modules = get_conditional_modules(model_name)
elif model_name == "googlenet":
    model = models.googlenet(pretrained=True).eval()
    all_layers = ["inception5b", "inception5a", "inception4e", "inception4d",
                  "inception4c", "inception4b", "inception4a", "inception3b",
                  "inception3a", "maxpool2"]
    conditional_modules = get_conditional_modules(model_name)
else:
    raise ValueError("Model not supported")

# %%
from captum.attr import IntegratedGradients, Saliency, GuidedBackprop, LRP, NoiseTunnel, GradientShap
from knockpy.knockoff_filter import KnockoffFilter

def masked_probing(intermediate, indices, netB, device = "cuda", batch_size = 128, reverse = False):
    if reverse == True:
        if isinstance(indices, torch.Tensor):
            indices = [i for i in range(intermediate.shape[1]) if i not in indices]
        else:
            indices = [i for i in range(intermediate.shape[1]) if i != indices]
            
    intermediate[:, indices] = 0
    out = batch_inference(netB, intermediate, batch_size=batch_size, device = device).detach().cpu()
    return out

# def get_integrated_gradients(
#     syn_data: torch.Tensor,
#     fm_id: int,
#     netB: torch.nn.Module,
# ) -> torch.Tensor:
#     integrated_gradients = IntegratedGradients(netB.to(device))
#     importance = []
#     for activation in syn_data:
#         activation = activation.unsqueeze(0).to(device)
#         activation.requires_grad = True
#         attributions_ig = integrated_gradients.attribute(
#             activation, baselines=activation * 0, target=fm_id
#         )
#         importance.append(
#             _wrapper(attributions_ig.detach().cpu())
#         )

#     return torch.sum(torch.abs(torch.cat(importance)), dim=0)

import random

def generate_unique_lists(N, M, list_length):
    if list_length > M + 1:
        raise ValueError(
            "list_length must be less than or equal to M + 1 to ensure all elements in the list are unique"
        )
    unique_lists = []
    for _ in range(N):
        unique_list = random.sample(range(M + 1), list_length)
        unique_lists.append(np.array(unique_list))
    return unique_lists

from tqdm import tqdm
def calculate_score(netB, data, num_nodes, device, batch_size, reverse):
    scores = []
    for node in tqdm(range(num_nodes)):
        masked_activation = masked_probing(
            data.clone(), 
            node, 
            netB, 
            device, 
            batch_size, 
            reverse
        )
        
        masked_top_imgs = torch.topk(
            masked_activation[:, test_node], num_top_imgs
        )[1].detach().cpu().numpy()
        
        score = accuracy(top_imgs, masked_top_imgs)
        scores.append(score)
    return scores

def calculate_score_multi_masking(netB, data, list_masking, device, batch_size, reverse):
    scores = []
    for indices in tqdm(list_masking):
        masked_activation = masked_probing(
            data.clone(), 
            torch.from_numpy(indices), 
            netB, 
            device, 
            batch_size, 
            reverse
        )
        masked_top_imgs = torch.topk(
            masked_activation[:, test_node], num_top_imgs
        )[1].detach().cpu().numpy()
        
        score = accuracy(top_imgs, masked_top_imgs)
        scores.append(score)
    return scores

def get_scores(
    syn_data: torch.Tensor,
    fm_id: int,
    netB: torch.nn.Module,
    method_name: str,
    activation_target: torch.Tensor | None = None,
    device: str = "cpu"
) -> torch.Tensor: 
    n_samples = 50
    
    if method_name == "integrated_gradients":
        attr = IntegratedGradients(netB.to(device))
        def attribute(x, target): # type: ignore
            return attr.attribute(x, baselines=activation * 0, target=target, n_steps=n_samples)
        
    elif method_name == "saliency":
        attr = Saliency(netB.to(device))
        def attribute(x, target): # type: ignore
            return attr.attribute(x, target=target)
        
    elif method_name == "smoothgrad":
        attr = IntegratedGradients(netB.to(device))
        def attribute(x, target): # type: ignore
            noise_tunnel = NoiseTunnel(attr)
            return noise_tunnel.attribute(x, nt_type='smoothgrad', target=target, n_steps=n_samples)
        
    elif method_name == "guided_backprop":
        attr = GuidedBackprop(netB.to(device))
        def attribute(x, target): # type: ignore
            return attr.attribute(x, target=target)
        
    elif method_name == "lrp":
        attr = LRP(netB.to(device))
        def attribute(x, target): # type: ignore
            return attr.attribute(x, target=target)
        
    elif method_name == "knockoff":
        assert activation_target is not None
        attr = KnockoffFilter(ksampler='gaussian', fstat='lasso')
        def attribute(x, target):
            x = _wrapper(x).detach().cpu().numpy()
            attr.forward(X=x, y=activation_target[:, target].detach().cpu().numpy(), fdr=1.0)
            return torch.from_numpy(attr.W)
        
    elif method_name == "gradient_shap":
        attr = GradientShap(netB.to(device))
        def attribute(x, target):
            return attr.attribute(x, baselines=activation * 0, target=target, n_samples=n_samples)
    else:
        raise ValueError("Method not supported")
    
    importance = []
    if method_name == "knockoff":
        attributions_ig = attribute(syn_data, fm_id) # type: ignore
        importance.append(attributions_ig)
        return torch.abs(torch.cat(importance))
    else:
        for activation in syn_data:
            activation = activation.unsqueeze(0).to(device)
            activation.requires_grad = True
            attributions_ig = attribute(activation, fm_id) # type: ignore
            importance.append(
                _wrapper(attributions_ig.detach().cpu()) # type: ignore
            )
        return torch.sum(torch.abs(torch.cat(importance)), dim=0)


# Default params
test_layers = [i for i in range(1, len(all_layers))]
reverse = False
num_top_imgs = 50

correlations = []
mean_time = []
for i, test_layer in enumerate(test_layers):
    correlations.append({length: [] for length in list_lengths})
    
    netA, _ = split_model(
        model, 
        all_layers[test_layer], 
        include_split_layer_in_first_part=True, 
        conditional_modules=conditional_modules
    )
    netB = Model_wrapper(
        get_sub_model(
            model, 
            all_layers[test_layer], 
            all_layers[test_layer-1], 
            True, 
            conditional_modules=conditional_modules
        )
    )
    
    print(all_layers[test_layer], all_layers[test_layer-1])
    
    for target_label in label_list:
        class_images, class_labels = load_model_data(data_dir, [target_label], model, device) 
        concept_data = get_crop_data(class_images)
        
        intermediate = batch_inference(netA, concept_data, device=device)
        del concept_data, class_images, class_labels
        
        num_node = intermediate.shape[1]

        activation = batch_inference(netB, intermediate, device=device)
        
        while True:
            # generate random test node
            test_node = random.randint(0, activation.shape[1]-1)
            
            top_imgs = torch.topk(
                activation[:, test_node], num_top_imgs
            )[1].detach().cpu().numpy()
            
            # importance = get_integrated_gradients(
            #     intermediate[torch.from_numpy(top_imgs)], test_node, netB
            # ).detach().cpu().numpy()
            
            valid_test_node = True
            
            num_tried = 0
            importance = [0]
            while np.sum(importance) == 0: # while attrib method (mostly knockoff) can't find non-zero importance
                num_tried += 1
                print("Running ", method_name)
                start_time = time.time()
                importance = get_scores(
                    intermediate[torch.from_numpy(top_imgs)], 
                    test_node, 
                    netB, 
                    method_name, 
                    activation_target=activation[top_imgs], 
                    device="cpu",
                ).detach().cpu().numpy() 
                end_time = time.time()
                print(method_name, ":", "Number of trials:", num_tried, "Execution time: ", end_time - start_time)
                
                if num_tried > 20:
                    valid_test_node = False # not a valid test node
                    print("Failed to get non-zero importance, try different test node")
                    break
            mean_time.append(end_time - start_time) # type: ignore
            
            if valid_test_node:  # this test_node is valid
                break
        
        for length in list_lengths:
            M = num_node-1
            
            if length > 1:
                list_masking = generate_unique_lists(N, M, length)
                
                multi_importances = []
                for indices in list_masking:
                    multi_importances.append(np.sum(importance[indices]))
                    
                multi_importances = np.array(multi_importances)
                scores = np.array(
                    calculate_score_multi_masking(netB, intermediate, list_masking, device, batch_size, reverse)
                )
                
            else:
                scores = np.array(
                    calculate_score(netB, intermediate, num_node, device, batch_size, reverse)
                )
                multi_importances = importance
                
            correlation_matrix = np.nan_to_num(np.corrcoef(scores, multi_importances), nan=0.0) # replace nan number if needed
            correlation_coefficient = correlation_matrix[0, 1]
            
            print("Layer:", all_layers[test_layer-1], " Length:", length, "Label:", target_label, " Correlation:", correlation_coefficient)
            correlations[i][length].append(correlation_coefficient)
            
    correlations[i] = {length: np.mean(values) for length, values in correlations[i].items()}
        
    torch.save(correlations, output_dir + f"correlations_{method_name}_{model_name}.pth")
    
print("Mean time: ", np.mean(mean_time))
torch.save(np.mean(mean_time), output_dir + f"mean_time_{method_name}_{model_name}.pth")
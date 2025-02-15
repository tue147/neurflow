# %%
import torch
import torch.nn as nn
from utils import split_model, get_sub_model, batch_inference, get_conditional_modules, load_model_data, _wrapper, Model_wrapper
from typing import List
import numpy as np
from tqdm import tqdm
import argparse
from captum.attr import IntegratedGradients

parser = argparse.ArgumentParser(description="Fidelity of critical neurons")

parser.add_argument('--data_dir', type=str, required=True, help='Directory to store the data')
parser.add_argument("--load_dir", type=str, default="./full_/", help="Directory to load previous runs from")
parser.add_argument("--model_name", type=str, default="googlenet", help="Model name to use")
parser.add_argument("--label_list", type=int, nargs='+', default=[0], help="List of labels (space-separated values)")
parser.add_argument("--list_tau", type=int, nargs='+', default=[16], help="Tau value")
parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for processing")
parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
parser.parse_args()

args = parser.parse_args()

data_dir = args.data_dir
load_dir = args.load_dir
model_name = args.model_name
label_list = args.label_list
list_tau = args.list_tau
batch_size = args.batch_size

device = args.device if torch.cuda.is_available() else "cpu"

from torchvision import models
if model_name == "resnet50":
    model = models.resnet50(pretrained=True).eval()
    conditional_modules = get_conditional_modules(model_name)
elif model_name == "googlenet":
    model = models.googlenet(pretrained=True).eval()
    conditional_modules = get_conditional_modules(model_name)
else:
    raise ValueError("Model not supported")

class_images, class_labels = load_model_data(data_dir, label_list, model, device)

def get_integrated_gradients(
    syn_data: torch.Tensor,
    fm_id: int,
    netB: torch.nn.Module,
    batch_size: int = 8
) -> torch.Tensor:
    integrated_gradients = IntegratedGradients(netB.to(device))
    importance = []
    for i in range(0, len(syn_data), batch_size):
        batch_data = syn_data[i:i + batch_size].to(device)
        batch_data.requires_grad = True
        attributions_ig = integrated_gradients.attribute(
            batch_data, baselines=batch_data * 0, target=fm_id, n_steps=50
        )
        importance.append(
            _wrapper(attributions_ig.detach().cpu())
        )
        batch_data.cpu().detach()
    return torch.sum(torch.abs(torch.cat(importance)), dim=0)

# %%
import random
def pruned_probing(
    layers, 
    from_layer, 
    model, 
    data, 
    device, 
    batch_size=batch_size, 
    method_name="NeurFlow",
    reverse = False, 
    baseline_value = 0, 
    num_features=-1
):
    netA, _ = split_model(
        model, 
        layers[-1], 
        include_split_layer_in_first_part=True, 
        conditional_modules=conditional_modules
    )
    start = 1
    end = len(layers) - 1
    # print(layers)
    intermediate = batch_inference(netA, data, device=device, batch_size=batch_size)
    for layer in range(start, end+1): 
        index = len(layers) - layer
        intermediate_layer = get_sub_model(
            model, 
            layers[index], 
            layers[index-1], 
            True, 
            conditional_modules=conditional_modules
        )
        mask = torch.zeros(intermediate.shape[1:]) if reverse else torch.ones(intermediate.shape[1:])
        
        if num_features < 0:
            indices = FW.get_critical_neurons(layer+from_layer-1)
        else:
            indices = FW.get_critical_feature_maps(
                layer+from_layer-1
            )[:min(num_features, len(FW.get_critical_neurons(layer+from_layer-1)))]
            
        if method_name == "NeurFlow":
            mask[indices] = 1 if reverse else 0
            
        elif method_name == "NeuronMCT":
            total = len(indices)
            _, netB = split_model(model, layers[index], include_split_layer_in_first_part=True, conditional_modules=conditional_modules)
            importance = get_integrated_gradients(intermediate, label, netB)

            indices = torch.topk(importance, total).indices

            
        baseline = torch.ones(intermediate.shape[1:]) * baseline_value
        baseline[mask.int()] = 0

        intermediate = batch_inference(
            intermediate_layer, 
            intermediate * mask + baseline,
            device=device, 
            batch_size=batch_size
        )
    return intermediate

def result(data, label, single = True, reverse = False, baseline_value = 0):
    mean_acc_random = []
    mean_acc = []
    for layer in range(0, len(layers)-2):
        # critical
        out = pruned_probing(
            layers[layer if single else 0:layer+2], 
            len(layers)-layer-2, 
            model, 
            data, 
            device, 
            method_name="NeurFlow", 
            reverse=reverse, 
            baseline_value=baseline_value
        )
        if single:
            classifier = get_sub_model(model, layers[layer], layers[0], True, conditional_modules=conditional_modules)
            out = batch_inference(classifier, out, device=device)
            
        _, predicted_labels = torch.max(out, 1)
        mean_acc.append(torch.sum(predicted_labels == label).item() / data.shape[0])
        
        # random
        out = pruned_probing(
            layers[layer if single else 0:layer+2], 
            len(layers)-layer-2, 
            model, 
            data, 
            device, 
            method_name="NeuronMCT", 
            reverse=reverse, 
            baseline_value=baseline_value
        )
        if single:
            classifier = get_sub_model(model, layers[layer], layers[0], True, conditional_modules=conditional_modules)
            out = batch_inference(classifier, out, device=device)
            
        _, predicted_labels = torch.max(out, 1)
        mean_acc_random.append(torch.sum(predicted_labels == label).item() / data.shape[0])
    return mean_acc, mean_acc_random

# %%
import matplotlib.pyplot as plt
from matplotlib import cm

list_rev = [False, True, False, True]
list_sin = [True, True, False, False]

default_keys = [None, None, None]  # Default placeholders
default_keys[:len(list_tau)] = list_tau  # Fill available values from list_tau
style_dict = {
    default_keys[0]: {
        'mean_marker': 's', 'mean_ls': '-',
        'random_marker': 's', 'random_ls': '--'
    } if default_keys[0] is not None else {},

    default_keys[1]: {
        'mean_marker': '^', 'mean_ls': '-',
        'random_marker': '^', 'random_ls': '--'
    } if default_keys[1] is not None else {},

    default_keys[2]: {
        'mean_marker': 'o', 'mean_ls': '-',
        'random_marker': 'o', 'random_ls': '--'
    } if default_keys[2] is not None else {}
}
# Remove empty dictionary keys (if any)
style_dict = {k: v for k, v in style_dict.items() if k is not None}

default_style = {
    'mean_marker': '2', 'mean_ls': '-.',
    'random_marker': 'p', 'random_ls': ':'
}
cmap_critical = cm.get_cmap('viridis_r', len(list_tau) + 1)
cmap_random = cm.get_cmap('magma_r', len(list_tau) * 2 + 1)
# available = ['default'] + plt.style.available
# with plt.style.context(available[17]):
fig, axes = plt.subplots(1, 4, figsize=(22, 5.5), sharey=True)

if model_name == "resnet50":
    name = "ResNet50"
    plot_layers = ["layer4.2", "", "layer4.0", "", "layer3.4", 
                        "", "layer3.2", "", "layer3.0"]
elif model_name == "googlenet":
    name = "GoogLeNet"
    plot_layers = ["inception5b", "", "inception4e", "",
                    "inception4c", "", "inception4a", "",
                    "inception3a"]
else:
    raise ValueError("Model not supported")

for i, (reverse, single) in enumerate(zip(list_rev, list_sin)):
    for tau_index, tau in enumerate(list_tau):
        mean_acc_runs = []
        mean_acc_random_runs = []
        
        directory = args.load_dir + f"{tau}/"
        for label in tqdm(label_list):
            path = directory + f"store_{model_name}_label{label}_tau{tau}.pth"
            
            store = torch.load(path, map_location="cpu", weights_only=False)
            FW = store["FW"]
            layers = store["layers"]
            del store
            
            temp_data = class_images[class_labels == label]
            mean_acc, mean_acc_random = result(temp_data, label, single=single, reverse=reverse)
            
            mean_acc_runs.append(mean_acc)
            mean_acc_random_runs.append(mean_acc_random)
        
        mean_acc_runs = np.array(mean_acc_runs)
        mean_acc_random_runs = np.array(mean_acc_random_runs)

        mean_acc_mean = mean_acc_runs.mean(axis=0)
        mean_acc_random_mean = mean_acc_random_runs.mean(axis=0)

        ploting_layer = layers[1:len(layers)-1] # type: ignore
        style = style_dict.get(tau, default_style)

        # Plot using the retrieved styles
        axes[i].plot(
            ploting_layer, mean_acc_mean, style['mean_marker'], linestyle=style['mean_ls'], markersize=10,
            color=cmap_critical(tau_index + 1)
        )
        axes[i].plot(
            ploting_layer, mean_acc_random_mean, style['random_marker'], linestyle=style['random_ls'], markersize=10, 
            color=cmap_random(tau_index + 1)
        )
    
    title_name = name + " "  
    title_name += "single layer" if single else "multi-layer"
    title_name += " retaining" if reverse else " masking"
    axes[i].set_title(title_name, fontsize=22)
    
    x = np.arange(len(ploting_layer)) # type: ignore
    if model_name == "resnet50":
        abbreviated_layers = [layer.replace('layer', '') for layer in plot_layers]
    else:
        abbreviated_layers = [layer.replace('inception', '') for layer in plot_layers]
    if model_name == "resnet50":
        axes[i].set_facecolor('#f5edf3') 
    if model_name == "googlenet":
        axes[i].set_facecolor('#e7f1e6')
    axes[i].set_xticks(x)
    axes[i].set_xticklabels(abbreviated_layers, fontsize=22)
    axes[i].tick_params(axis='y', labelsize=22)
    axes[i].grid(True, axis='y', linestyle='--', alpha=0.8)

fig.text(0.5, 0.01, 'Layers', ha='center', fontsize=24)
fig.text(0.01, 0.5, 'Accuracy', va='center', rotation='vertical', fontsize=24)
axes[0].set_ylim(-0.1, 1.1)
create_legend = []
for tau in list_tau:
    create_legend += [f'$\\tau = {tau}$-CC', f'$\\tau = {tau}$-Random']
fig.legend(create_legend, loc='upper center', bbox_to_anchor=(0.5, 1), ncol=len(create_legend), fontsize=20)

plt.tight_layout(rect=[0.02, 0.05, 1, 0.90]) # type: ignore
plt.show()

string_tau = "_".join([str(i) for i in list_tau])
plt.savefig(f"{model_name}_fidelity_{string_tau}.png")
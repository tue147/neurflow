# %%
import torch
import torchvision.models as models
from torchvision import transforms
from torch import nn
import random
import numpy as np
from matplotlib import pyplot as plt
from ...utils import show, split_model, get_sub_model, batch_inference, get_conditional_modules, load_model_data
import argparse
    
parser = argparse.ArgumentParser(description="NeurFlow Setups")
parser.add_argument('--label', type=int, required=True, help='Label index for filtering')
parser.add_argument('--model', type=str, required=True, default="googlenet", help='Model to use')
parser.add_argument('--output_dir', type=str, required=True, help='Directory to store the results')
parser.add_argument('--data_dir', type=str, required=True, help='Directory to store the data')
parser.add_argument('--tau', type=int, required=True, help='The tau value')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for the forward pass')
parser.add_argument('--batch_size_ig', type=int, default=8, help='Batch size for the integrated gradients')
parser.add_argument('--device', type=str, default="cuda:0", help='Device to use')
args = parser.parse_args()

device = args.device if torch.cuda.is_available() else "cpu"

# %%
from torchvision import models
model_name = args.model
if model_name == "resnet50":
    model = models.resnet50(pretrained=True).eval()
    all_layers = ["fc", "layer4.2", "layer4.1", "layer4.0", "layer3.5", 
                "layer3.4", "layer3.3", "layer3.2", "layer3.1", "layer3.0", 
                "layer2.3", "layer2.2", "layer2.1", "layer2.0", "layer1.2",
                "layer1.1", "layer1.0", "conv1"]
    conditional_modules = get_conditional_modules(model_name)
elif model_name == "googlenet":
    model = models.googlenet(pretrained=True).eval()
    all_layers = ["fc", "inception5b", "inception5a", "inception4e", "inception4d",
                  "inception4c", "inception4b", "inception4a", "inception3b",
                  "inception3a", "maxpool2"]
    conditional_modules = get_conditional_modules(model_name)
else:
    raise ValueError("Model not supported")

# %% Load Data
target_label = args.label
all_images, all_labels = load_model_data(args.data_dir, [target_label], model, device)

# %%
from NeurFlow import Framework

for i in range(1, len(all_layers)-1):
    layers = ["fc"] + all_layers[i:i+2]
    print(layers)
    start = 1
    end = len(layers) - 1
    num_max_top_neu = [16 for _ in layers]
    FW = Framework(start, end, all_images, target_label, device, granularities=[1, 0.5, 0.25])
    for layer in range(end, start-1, -1):
        index = len(layers) - layer
        netA, _ = split_model(model, layers[index], True, conditional_modules)
        netB = get_sub_model(model, layers[index], layers[index-1], True, conditional_modules)
        print("Layer: ", layer, " ", layers[index])
        FW.fit(
            layer, 
            netA, 
            netB, 
            top_images_per_neuron = 50, 
            max_top_neurons = 50 if layer==end else args.tau,
            acceptable_error_range=-1,
            batch_size_ig=args.batch_size_ig,
            batch_size=args.batch_size,
        )
    
    FW.concept_data = None  # type: ignore
    store = {
        "FW": FW,
        "labels": target_label,
        "layers": layers,
    }
    torch.save(store, f"./{args.output_dir}/store_{model_name}_label{target_label}_layer_{layers[1]}_tau_{args.tau}.pth")
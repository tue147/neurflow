# %%
import torch
from utils import split_model, get_sub_model, batch_inference, accuracy, Model_wrapper, load_model_data, get_conditional_modules, get_crop_data
import numpy as np

# %%
from torchvision import models
import warnings
from math import ceil
from matplotlib import pyplot as plt
from random import random
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description="Optimality of critical neurons")

parser.add_argument('--data_dir', type=str, required=True, help='Directory to store the data')
parser.add_argument("--load_dir", type=str, default="./run_each_layer/", help="Directory to load previous runs from")
parser.add_argument("--model_name", type=str, default="googlenet", help="Model name to use")
parser.add_argument("--label", type=int, default=0, help="List of labels (space-separated values)")
parser.add_argument("--tau", type=int, default=50, help="Tau value")
parser.add_argument("--num_node_test", type=int, default=5, help="Number of nodes to test")
parser.add_argument("--num_random_comb", type=int, default=10, help="Number of random combinations")
parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for processing")
parser.add_argument("--output_dir", type=str, default="./minimization/", help="Directory to save output")
parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
parser.parse_args()

args = parser.parse_args()

data_dir = args.data_dir
load_dir = args.load_dir
model_name = args.model_name
target_label = args.label
tau = args.tau
num_node_test = args.num_node_test
num_random_comb = args.num_random_comb
batch_size = args.batch_size
output_dir = args.output_dir

device = args.device if torch.cuda.is_available() else "cpu"

def masked_probing(data, indices, net, device = "cuda", batch_size = 256, reverse = True):
    data = data.to("cpu")
    if reverse:
        reverse_indices = [i for i in range(data.shape[1]) if i not in indices]
        data[:, reverse_indices] = 0
    else:
        data[:, indices] = 0    
    return batch_inference(net, data, batch_size=batch_size, device = device)

def probing(data, net, device = "cuda", batch_size = 128):
    return batch_inference(net, data, batch_size=batch_size, device = device)

def top_img(probed, num_img = 5):
    _, indices = torch.topk(probed, num_img)
    return indices

def generate_random_comb(n, k, not_in = []):
    '''
    Generate a random combination of k elements from n elements.
    '''
    indices = []
    for i in range(k):
        index = int(random() * n)
        while index in indices or index in not_in:
            index = int(random() * n)
        indices.append(index)
    return indices

def generate_random_comb_from_list(k, list_indices = []):
    '''
    Generate a random combination of k elements from n elements.
    '''
    assert len(list_indices) >= k
    if k == len(list_indices):
        return list_indices
    indices = []
    for i in range(k):
        index = int(random() * len(list_indices))
        while index in indices:
            index = int(random() * len(list_indices))
        indices.append(list_indices[index])
    return indices

from torchvision import models
if model_name == "resnet50":
    model = models.resnet50(pretrained=True).eval()
    all_layers = ["layer4.2", "layer4.1", "layer4.0", "layer3.5", "layer3.4",
                  "layer3.3", "layer3.2", "layer3.1", "layer3.0", "layer2.3",
                  "layer2.2", "layer2.1", "layer2.0", "layer1.2", "layer1.1", "layer1.0"]
    conditional_modules = get_conditional_modules(model_name)
elif model_name == "googlenet":
    model = models.googlenet(pretrained=True).eval()
    all_layers = ["inception5b", "inception5a", "inception4e", "inception4d",
                  "inception4c", "inception4b", "inception4a", "inception3b",
                  "inception3a"]
    conditional_modules = get_conditional_modules(model_name)
else:
    raise ValueError("Model not supported")

# %%
class_images, class_labels = load_model_data(data_dir, [target_label], model, device) 
concept_data = get_crop_data(class_images)

mean_accs = []
for layer in all_layers:
    str_labels = "_".join([str(i) for i in [target_label]])
    path = load_dir + f"/store_{model_name}_label{str_labels}_layer_{layer}_tau_{tau}.pth"

    store = torch.load(path, map_location="cpu", weights_only=False)
    FW = store["FW"]
    layers = store["layers"]
    del store
    print(layers[1])
    
    test_layer = 1 # run_each_layer.py only stores 3 layers (fc, and 2 intermediate layers), so we want to extract the "layer=1"
    num_imgs = 50 # default param
    
    dict_indices = {}
    for fm in FW.get_critical_neurons(test_layer)[:min(num_node_test, len(FW.get_critical_neurons(test_layer)))]:
        dict_indices[fm] = list({
            k for dict in FW.get_critical_neurons(test_layer, fm).values() for k in dict.keys()
        })

    netA, _ = split_model(
        model, 
        layers[-test_layer], 
        include_split_layer_in_first_part=True, 
        conditional_modules=conditional_modules
    )
    netB = Model_wrapper(
        get_sub_model(
            model, 
            layers[-test_layer], 
            layers[-test_layer-1], 
            True, 
            conditional_modules=conditional_modules
        )
    )
    hidden = probing(concept_data, netA, device = device, batch_size=batch_size)
    activation = batch_inference(netB, hidden, device=device)
    
    all_acc = {}
    all_ori_acc = {}
    
    for fm, indices in dict_indices.items():
        print(f"Feature map {fm}")
        
        top_imgs = torch.topk(activation[:, fm], num_imgs)[1].detach().cpu().numpy()
        original_masked = masked_probing(
            hidden.clone(), 
            indices, 
            netB, 
            device = device, 
            reverse=False, 
            batch_size=batch_size
        )[:, fm].detach().cpu()
        
        top_img_indices_ori_masked = top_img(original_masked, num_img=num_imgs).numpy()
        ori_acc = accuracy(top_imgs, top_img_indices_ori_masked)
        
        all_ori_acc[fm] = ori_acc
        all_acc[fm] = []
        
        for j in tqdm(range(num_random_comb)):
            # random combination
            test_comb = generate_random_comb(hidden.shape[1], tau, [])
            
            masked = masked_probing(
                hidden.clone(), 
                test_comb, 
                netB, 
                device = device, 
                reverse=False, 
                batch_size=batch_size
            )[:, fm].detach().cpu()
            top_img_indices_masked = top_img(masked, num_img=num_imgs).numpy()
            
            acc = accuracy(top_imgs, top_img_indices_masked)
            all_acc[fm].append(acc)

    mean_below_ori = []
    mean_min_acc = []
    mean_acc = []
    for fm, list_acc in all_acc.items():
        print("=====================================")
        print(f"Feature map {fm}")
        print(f"Original accuracy: {all_ori_acc[fm]}")
        print(f"Mean accuracy: {np.mean(list_acc)}")
        print(f"Std accuracy: {np.std(list_acc)}")
        print("")
        below_ori = []
        for acc in list_acc:
            if acc < all_ori_acc[fm]:
                below_ori.append(acc)
        mean_below_ori.append(np.mean(below_ori) - all_ori_acc[fm] if len(below_ori) > 0 else 0)
        mean_min_acc.append(min(list_acc) - all_ori_acc[fm])
        mean_acc.append(np.mean(list_acc) - all_ori_acc[fm])
        print("Below original accuracy", np.mean(below_ori)- all_ori_acc[fm] if len(below_ori) > 0 else 0) 
        print("Min accuracy", min(list_acc) - all_ori_acc[fm])

    print("Mean below original accuracy", np.mean(mean_below_ori))
    print("Mean min accuracy compare to original", np.mean(mean_min_acc))
    print("Mean accuracy compare to original", np.mean(mean_acc))
    
    torch.save([mean_acc, mean_below_ori, mean_min_acc, all_acc, all_ori_acc], output_dir + f"/store_{model_name}_{layers[-test_layer-1]}_{str_labels}_tau_{tau}.pth")
# %%

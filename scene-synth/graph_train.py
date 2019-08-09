import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.misc as m
from torchvision import datasets, transforms
from models.graph import *
import random
from data import *
import pickle
import utils

parser = argparse.ArgumentParser(description='Location Training with Auxillary Tasks')
parser.add_argument('--room-type', type=str, default="", metavar='S')
parser.add_argument('--num-epochs', type=int, default=120, metavar='N')
args = parser.parse_args()

if len(args.room_type) > 0:
    data_dir = f"data/{args.room_type}"
    save_dir = f"results_{args.room_type}"
else:
    data_dir = ""
    save_dir = ""
    #save_dir = "sophon/bedroom"

valid_set_size = 160
batch_size = 8 #
dataset_size = 5700
dataset_size = int(dataset_size / batch_size) * batch_size
print(dataset_size)

test_epoch = 65
test_debug = False
test_with_gt = True
show_img = False
if not test_debug:
    test_with_gt = False
    show_img = False

sym_class_names = ["No", "Radial", "FB", "LR", "R2", "R4", "C1", "C2"]

if test_epoch >= 0:
    load_dir = save_dir
    save_dir = ""

utils.ensuredir(save_dir)

logfile = open(f"{save_dir}/log.txt", 'w')
def LOG(msg):
    print(msg)
    logfile.write(msg + '\n')
    logfile.flush()


cuda = False

shuffle_nodes = True
shuffle_edges = True
shuffle_nodes_p = 1

include_walls = True
include_arch = False
#order_nodes = True
print_progress = False

categories = ObjectCategories().all_non_arch_categories_importance_order(".",data_dir)
num_cat = len(categories)
arch_categories = ["wall"] + ObjectCategories().all_arch_categories()
num_arch_cat = len(arch_categories) 
cat_to_index = {categories[i]:i for i in range(len(categories))}
arch_cat_to_index = {arch_categories[i]:i for i in range(len(arch_categories))}

#Configs used for final version
config = {
    "shuffle_nodes": shuffle_nodes,
    "shuffle_edges": shuffle_edges,
    "include_walls": include_walls,
    "hidden_size": 384,
    "num_cat": num_cat,
    "num_arch_cat": num_arch_cat,
    "categories": categories,
    "arch_categories": arch_categories,
    "autoregressive": False,
    "init_with_graph_representation": True,
    "decision_layer": Linear3,
    "initializing_layer": Linear3,
    "propagation_layer": Linear3,
    "aggregation_layer": Linear3,
    #"dropout_p": 0.2,
    "element_wise_loss_node": False,
    "predict_edge_type_first": False,
    "choose_node_graph_vector": True,
    "node_and_type_together": True,
    "everything_together": True,
    "include_one_hot": True,
    "cuda": cuda,
    "separate_wall_edges_step": True,
    #"node_vector_activation_f": nn.ReLU(),
    #"restore_target": {"ae": "an", "node": "an", "et": None},
    #"restore_target": {"ae": "initial", "node": None, "et": None},
    "rounds_of_propagation_dict": {"an": 3, "ae": 3, "node": 3, "et": 3},
    #"temp_dict": {"an": 4, "ae": 1, "node": 4, "et": 1}
}

if test_epoch >= 0:
    with open(f"{load_dir}/config.pkl", 'rb') as f:
        config = pickle.load(f)
        print(config)
else:
    with open(f"{save_dir}/config.pkl", 'wb') as f:
        pickle.dump(config, f, pickle.HIGHEST_PROTOCOL)

for (key, value) in config.items():
    setattr(GraphNetConfig, key, value)
GraphNetConfig.compute_derived_attributes()

#a = layer_name_to_class["1"](5,1)
print(GraphNetConfig.__dict__)
model = GraphNet()
if cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=0.0001, eps=0.001)

def train():
    model.train()
    total_losses = {}
    total_losses["an"] = torch.zeros(1) #add node
    total_losses["ae"] = torch.zeros(1) #add edge
    total_losses["node"] = torch.zeros(1) #choose node
    total_losses["et"] = torch.zeros(1)
    total_losses["sym"] = torch.zeros(1)
    if cuda:
        for key in total_losses.keys():
            total_losses[key] = total_losses[key].cuda()

    for i in range(1000):
        if print_progress:
            print(i)
        debug = False
        optimizer.zero_grad()

        losses = model.train_step(random_graph(), debug=debug)

        loss = sum(losses.values())
        loss.backward()
        optimizer.step()
        n = len(model.gt_nodes)
        
        for key in losses.keys():
            total_losses[key] += losses[key]/n
    
    LOG("Losses:")   
    for (key, value) in total_losses.items():
        LOG(f"    {key}: {(value/1000)[0].data.cpu().numpy()}")
    
for i in range(args.num_epochs):
    torch.save(model.state_dict(), f"{save_dir}/{i}.pt")
    LOG(f'=========================== Epoch {i} ===========================')
    #sample(i)
    train()

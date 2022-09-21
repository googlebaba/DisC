import torch
import torch.nn as nn
import numpy as np
import random
import argparse
import pdb
import os
from tqdm import tqdm
import dgl
from dgl.nn.pytorch import GraphConv
import torch.nn.utils.prune as prune

device = torch.device('cuda')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)



def print_args(args, str_num=80):
    for arg, val in args.__dict__.items():
        print(arg + '.' * (str_num - len(arg) - len(str(val))) + str(val))
    print()
    
def parser_loader():
    parser = argparse.ArgumentParser(description='GNN baselines on ogbg-ppa data with Pytorch Geometrics')
    parser.add_argument('--pa', type=float, default=0)
    parser.add_argument('--pw', type=float, default=0)

    parser.add_argument('--irm_lambda', type=float, default=0)

    parser.add_argument('--lambda_dis', type=float, default=1.)

    parser.add_argument('--lambda_swap', type=float, default=1.)

    parser.add_argument('--q', type=float, default=0.7)

    parser.add_argument('--masker_lr', type=float, default=0.01)
    parser.add_argument('--seed', type=int,help="Please give a value for seed")
    parser.add_argument('--config', help="Please give a config.json file with training/model/data/param details")
    parser.add_argument('--epochs', type=int, help="Please give a value for epochs")
    parser.add_argument('--mask_epochs', type=int, help="Please give a value for epochs")

    parser.add_argument('--swap_epochs', type=int, help="Please give a value for epochs")
    parser.add_argument('--eval_epochs', type=int, help="Please give a value for epochs")

    parser.add_argument('--use_mask', type=int, help="Please give a value for epochs")
    parser.add_argument('--gpu_id', help="Please give a value for gpu id")
    parser.add_argument('--model', help="Please give a value for model name")
    parser.add_argument('--dataset', help="Please give a value for dataset name")

    parser.add_argument('--data_dir', help="Please give a value for dataset dir")
    parser.add_argument('--out_dir', help="Please give a value for out_dir")
    return parser



"""
    GPU Setup
"""
def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:',torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device

def prRed(skk): print("\033[91m{}\033[00m".format(skk))
def prGreen(skk): print("\033[92m{}\033[00m".format(skk))
def prYellow(skk): print("\033[93m{}\033[00m".format(skk))

def pruning_batch_data_from_mask(data_list, batch_labels, data_mask, args):
    offset = 0
    nodes_offset = 0
    new_data_list = []
    label_list = []
    for data, label in zip(data_list, batch_labels):
        num_edges = data.number_of_edges()
        edge_score = data_mask[offset:offset + num_edges]
        mean_score = torch.mean(edge_score)
        sorted_score, index = torch.sort(edge_score.view(-1))

        prune_num_edges = int(num_edges * 0.8)
        #_, index = torch.sort(edge_score.view(-1))
        prune_index = index[:prune_num_edges]
        data.remove_edges(prune_index)
        #data.edata['mask'] = edge_score.cpu()
        isolated_nodes = ((data.in_degrees() == 0) & (data.out_degrees() == 0)).nonzero().squeeze(1)
        data.remove_nodes(isolated_nodes)
        new_data_list.append(data)
        label_list.append(label)
        offset += num_edges
        
    return new_data_list, label_list

def pruning_batch_data_from_mask_cp(data_list, batch_labels, data_mask, args):
    offset = 0
    nodes_offset = 0
    new_data_list = []
    label_list = []
    for data, label in zip(data_list, batch_labels):
        num_edges = data.number_of_edges()
        edge_score = data_mask[offset:offset + num_edges]
        mean_score = torch.mean(edge_score)
        sorted_score, index = torch.sort(edge_score.view(-1))

        prune_num_edges = int(torch.sum(torch.lt(edge_score, mean_score)))
        #_, index = torch.sort(edge_score.view(-1))
        prune_index = index[:prune_num_edges]
        data.remove_edges(prune_index)
        #data.edata['mask'] = edge_score.cpu()
        isolated_nodes = ((data.in_degrees() == 0) & (data.out_degrees() == 0)).nonzero().squeeze(1)
        data.remove_nodes(isolated_nodes)
        new_data_list.append(data)
        label_list.append(label)
        offset += num_edges
        
    return new_data_list, label_list


class DGLFormDataset(torch.utils.data.Dataset):
    """
        DGLFormDataset wrapping graph list and label list as per pytorch Dataset.
        *lists (list): lists of 'graphs' and 'labels' with same len().
    """
    def __init__(self, *lists):
        assert all(len(lists[0]) == len(li) for li in lists)
        self.lists = lists
        self.graph_lists = lists[0]
        self.graph_labels = lists[1]

    def __getitem__(self, index):
        return tuple(li[index] for li in self.lists)

    def __len__(self):
        return len(self.lists[0])

def masker_pruning_dataset(dataloader, masker, device, args):
    data_list = []
    label_list = []
    with torch.no_grad():
        for iter, (batch_graphs, batch_labels) in enumerate(dataloader):

            batch_graphs = dgl.batch(batch_graphs).to(device)
            batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
            #batch_e = batch_graphs.edata['feat'].to(device)
            batch_e = None
            data_mask, node_masker = masker(batch_graphs, batch_x, batch_e)
            batch_graphs = dgl.unbatch(batch_graphs)
            batch_graphs_pruned, label_pruned = pruning_batch_data_from_mask(batch_graphs, batch_labels, data_mask, args)
            data_list += batch_graphs_pruned
            label_list += label_pruned        
    #pruned_data = DGLFormDataset(data_list, label_list)
    return data_list, label_list

def see_zero_rate(model):
    sum_list = 0
    zero_sum = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            sum_list = sum_list + float(m.weight.nelement())
            zero_sum = zero_sum + float(torch.sum(m.weight == 0))     
    print('INFO: Weight Sparsity [{:.4f}%] '.format(100 * (zero_sum / sum_list)))
    return zero_sum / sum_list

def extract_mask(model):

    model_dict = model.state_dict()
    new_dict = {}
    for key in model_dict.keys():
        if 'mask' in key:
            new_dict[key] = model_dict[key]
    return new_dict

def pruning_model_by_mask(model, mask_dict):

    module_to_prune = []
    mask_to_prune = []
    module_to_prune.append(model.embedding_h)
    mask_to_prune.append(mask_dict['embedding_h.weight_mask'])
    module_to_prune.append(model.layers[0].apply_mod.linear)
    mask_to_prune.append(mask_dict['layers.0.apply_mod.linear.weight_mask'])
    module_to_prune.append(model.layers[1].apply_mod.linear)
    mask_to_prune.append(mask_dict['layers.1.apply_mod.linear.weight_mask'])
    #module_to_prune.append(model.layers[2].apply_mod.linear)
    #mask_to_prune.append(mask_dict['layers.2.apply_mod.linear.weight_mask'])
    #module_to_prune.append(model.layers[3].apply_mod.linear)
    #mask_to_prune.append(mask_dict['layers.3.apply_mod.linear.weight_mask'])
    #module_to_prune.append(model.MLP_layer.FC_layers[0])
    #mask_to_prune.append(mask_dict['MLP_layer.FC_layers.0.weight_mask'])
    #module_to_prune.append(model.MLP_layer.FC_layers[1])
    #mask_to_prune.append(mask_dict['MLP_layer.FC_layers.1.weight_mask'])
    print("linear", model.MLP_layer)
    module_to_prune.append(model.MLP_layer)
    mask_to_prune.append(mask_dict['MLP_layer.weight_mask'])
    for ii in range(len(module_to_prune)):
        prune.CustomFromMask.apply(module_to_prune[ii], 'weight', mask=mask_to_prune[ii])


def pruning_model(model, px, random=False):

    if px == 0:
        pass
    else:
        parameters_to_prune =[]
        for m in model.modules():
            if isinstance(m, nn.Linear):
                parameters_to_prune.append((m,'weight'))
                print(m)
        parameters_to_prune = tuple(parameters_to_prune)
        if random:
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.RandomUnstructured,
                amount=px,
            )
        else:
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=px,
            )



def print_pruning_percent(dataset_ori, dataset_pru, str1):

    ori_all = 0.0
    pru_all = 0.0

    for data_ori, data_pru in zip(dataset_ori, dataset_pru):
        #print(str1, data_ori[1], data_ori[2])
        ori = dgl.batch(data_ori[0]).number_of_edges()
        pru = dgl.batch(data_pru[0]).number_of_edges()
        ori_all += ori
        pru_all += pru
    
    sp = 1 - pru_all / ori_all
    # print('INFO: Dataset Sparsity [{:.4f}%] '.format(100 * sp))
    return sp


def plot_mask(data_mask):

    a = (data_mask <= 0.2).sum()
    b = (data_mask <= 0.4).sum()
    c = (data_mask <= 0.6).sum()
    d = (data_mask <= 0.8).sum()
    e = (data_mask <= 1.0).sum()
    a, b, c, d, e = float(a), float(b), float(c), float(d), float(e)

    a1 = a / e         # (0.0 - 0.2)
    b1 = (b - a) / e   # (0.2 - 0.4)
    c1 = (c - b) / e   # (0.4 - 0.6)
    d1 = (d - c) / e   # (0.6 - 0.8)
    e1 = (e - d) / e   # (0.8 - 1.0)

    return [a1, b1, c1, d1, e1]


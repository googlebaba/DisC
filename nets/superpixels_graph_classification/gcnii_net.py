import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

"""
    GCN: Graph Convolutional Networks
    Thomas N. Kipf, Max Welling, Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)
    http://arxiv.org/abs/1609.02907
"""
from layers.gcn_layer import GCNLayer
from layers.mlp_readout_layer import MLPReadout
import pdb
import torch.autograd as autograd
import numpy as np

import math

from layers.gcnii_layer import GCNIILayer
#from layers.gcnii_variant_layer import GCNIIVariantLayer
import torch.nn as nn
import torch.nn.functional as F

def cal_gain(fun, param=None):
    gain = 1
    if fun is F.sigmoid:
        gain = nn.init.calculate_gain('sigmoid')
    if fun is F.tanh:
        gain = nn.init.calculate_gain('tanh')
    if fun is F.relu:
        gain = nn.init.calculate_gain('relu')
    if fun is F.leaky_relu:
        gain = nn.init.calculate_gain('leaky_relu', param)
    return gain

class GCNIINet(nn.Module):
    def __init__(self, net_params, variant=False):
        super(GCNIINet, self).__init__()
        self.convs = nn.ModuleList()
        num_feats = net_params['in_dim']
        out_dim = net_params['out_dim']
        num_classes = net_params['n_classes']
        num_hidden = net_params['hidden_dim']
        dropout = net_params['dropout']
        alpha = 0.1
        activation=F.relu
        graph_norm=True
        bias=True
        lamda=0.5
        num_layers = net_params['L']

        for i in range(num_layers):
            beta = math.log(lamda/(i+1)+1)
            if variant:
                self.convs.append(GCNIIVariantLayer(num_hidden, num_hidden, bias, activation,
                                             graph_norm, alpha, beta))
            else:
                self.convs.append(GCNIILayer(num_hidden, num_hidden, bias, activation,
                                             graph_norm, alpha, beta))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(num_feats, num_hidden))

        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.activation = activation
        self.dropout = dropout

        self.MLP_layer = nn.Linear(out_dim*2, num_classes)        
        self.reset_parameters()

    def reset_parameters(self):
        gain = cal_gain(self.activation)
        nn.init.xavier_uniform_(self.fcs[0].weight, gain=gain)
        if self.fcs[0].bias is not None:
            nn.init.zeros_(self.fcs[0].bias)
        nn.init.xavier_uniform_(self.fcs[-1].weight)
        if self.fcs[-1].bias is not None:
            nn.init.zeros_(self.fcs[-1].bias)
    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss(reduction='none')
        loss = criterion(pred, label)
        return loss

    def forward(self, graph, features, edge, mask, data_mask):
        h0 = F.dropout(features, self.dropout, self.training)
        h0 = self.activation(self.fcs[0](h0))
        h = h0
        for con in self.convs:
            h = F.dropout(h, self.dropout, self.training)
            h = con(graph, h, h0, mask)
        h = F.dropout(h, self.dropout, self.training)
        #h = self.fcs[-1](h)
        graph.ndata['h'] = h
        hg = dgl.mean_nodes(graph, 'h')
        return hg


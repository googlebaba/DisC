import torch
import torch.nn as nn
import math
import time
import pruning
from train.metrics import accuracy_MNIST_CIFAR as accuracy
import pdb
import numpy as np
import dgl
from utils import GeneralizedCELoss


    
def loss_dependence(emb1, emb2, dim):
    R = torch.eye(dim).cuda() - (1/dim) * torch.ones(dim, dim).cuda()
    K1 = torch.mm(emb1, emb1.t())
    K2 = torch.mm(emb2, emb2.t())
    RK1 = torch.mm(R, K1)
    RK2 = torch.mm(R, K2)
    HSIC = torch.trace(torch.mm(RK1, RK2))
    return HSIC


def train_model_and_masker(model_c, masker_c,  model_b, masker_b, optimizer_c, optimizer_b, sample_loss_ema_c, sample_loss_ema_b, device, data_loader, epoch, args):
    bias_criterion = GeneralizedCELoss(q=args.q)
    model_b.train()
    masker_b.train()
    model_c.train()
    masker_c.train()
    
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    mask_distribution = []
    for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
        batch_graphs = dgl.batch(batch_graphs).to(device)

        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        #batch_e = batch_graphs.edata['feat'].to(device)
        batch_labels = batch_labels.to(device)
        optimizer_c.zero_grad()
        optimizer_b.zero_grad()
        data_maskers = []
        data_mask_c, data_mask_node_c = masker_c(batch_graphs, batch_x, None)
        mask_dis = pruning.plot_mask(data_mask_c)
        mask_distribution.append(mask_dis)
        data_mask_b = 1-data_mask_c
        data_mask_node_b = 1 - data_mask_node_c
        
        if not args.use_mask:
            data_mask_c = None
            data_mask_b = None
        batch_scores_c = model_c(batch_graphs, batch_x, None, data_mask_c, data_mask_node_c)
        batch_scores_b = model_b(batch_graphs, batch_x, None, data_mask_b, data_mask_node_b)
        independ_loss = loss_dependence(batch_scores_c, batch_scores_b, batch_labels.size(0))
        z_c = torch.cat((batch_scores_c, batch_scores_b.detach()), dim=1)
        z_b = torch.cat((batch_scores_c.detach(), batch_scores_b), dim=1)
        batch_scores_c1 = model_c.MLP_layer(z_c)

        batch_scores_b1 = model_b.MLP_layer(z_b)
        
        loss_c = model_c.loss(batch_scores_c1, batch_labels).detach()

        loss_b = model_b.loss(batch_scores_b1, batch_labels).detach()
        
        loss_weight = loss_b / (loss_b + loss_c + 1e-8)

       # loss_weight = loss_dis_align / (loss_dis_align + loss_dis_conflict + 1e-8)
        #print("weights:", loss_weight)
        loss_dis_conflict = model_c.loss(batch_scores_c1, batch_labels) * loss_weight.to(device)
        #print("******************", loss_dis_conflict, loss_weight)
        loss_dis_align = bias_criterion(batch_scores_b1, batch_labels)
        if epoch > args.swap_epochs:
            indices = np.random.permutation(batch_scores_b.size(0))
            z_b_swap = batch_scores_b[indices]         # z tilde
            label_swap = batch_labels[indices]     # y tilde
            # Prediction using z_swap=[z_l, z_b tilde]
            # Again, gradients of z_b tilde are not backpropagated to z_l (and vice versa) in order to guarantee disentanglement of representation.
            z_mix_conflict = torch.cat((batch_scores_c, z_b_swap.detach()), dim=1)
            z_mix_align = torch.cat((batch_scores_c.detach(), z_b_swap), dim=1)
            # Prediction using z_swap
            pred_mix_conflict = model_c.MLP_layer(z_mix_conflict)
            pred_mix_align = model_b.MLP_layer(z_mix_align)
            loss_swap_conflict = model_c.loss(pred_mix_conflict, batch_labels) * loss_weight.to(device)     # Eq.3 W(z)CE(C_i(z_swap),y)
            loss_swap_align = bias_criterion(pred_mix_align, label_swap)                               # Eq.3 GCE(C_b(z_swap),y tilde)
            lambda_swap = args.lambda_swap                                         # Eq.3 lambda_swap_b
        else:
            # before feature-level augmentation
            loss_swap_conflict = torch.tensor([0]).float()
            loss_swap_align = torch.tensor([0]).float()
            lambda_swap = 0

        loss_swap = loss_swap_conflict.mean() + args.lambda_dis*loss_swap_align.mean()
        loss_dis = loss_dis_conflict.mean() + args.lambda_dis*loss_dis_align.mean()
        loss = loss_dis + lambda_swap * loss_swap #+ 1e-10*independ_loss #+ 0.00001*l1_loss
        loss.backward()
        #for name, parms in masker_c.named_parameters():
        #    print('-->name:', name, '-->grad_requirs:', parms.requires_grad, ' -->grad_value:', parms.grad)
        optimizer_b.step()
        optimizer_c.step()

        epoch_loss += loss.detach().item()
        epoch_train_acc += accuracy(batch_scores_c1, batch_labels)
        nb_data += batch_labels.size(0)
    
        #if iter % 40 == 0:
        #    print('-'*120)
        #    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' | ' +
        #            'Epoch: [{}/{}] Iter: [{}/{}] Loss:[{:.4f}] Train Acc:[{:.2f}] LR1:[{:.6f}] LR2:[{:.6f}] | 0-0.2:[{:.2f}%] 0.2-0.4:[{:.2f}%] 0.4-0.6:[{:.2f}%] 0.6-0.8:[{:.2f}%] 0.8-1.0:[{:.2f}%]'
        #            .format(epoch + 1, 
        #                    args.mask_epochs, 
        #                    iter, 
        #                    len(data_loader), 
        #                    epoch_loss / (iter + 1), 
        #                    epoch_train_acc / nb_data * 100,
        #                    optimizer_c.param_groups[0]['lr'],
        #                    optimizer_c.param_groups[1]['lr'],
        #                    mask_dis[0] * 100,
        #                    mask_dis[1] * 100,
        #                    mask_dis[2] * 100,
        #                    mask_dis[3] * 100,
        #                    mask_dis[4] * 100))

    mask_distribution = torch.tensor(mask_distribution).mean(dim=0)
    epoch_loss /= (iter + 1)
    epoch_train_acc /= nb_data

    return epoch_loss, epoch_train_acc, optimizer_c, mask_distribution

         

    

def eval_acc_with_mask(model_c, masker_c, model_b, device, data_loader, epoch, args, binary=False, val=False):

    model_c.eval()
    masker_c.eval()
    model_b.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0
    with torch.no_grad():
        for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
            batch_graphs = dgl.batch(batch_graphs).to(device)
            batch_x = batch_graphs.ndata['feat'].to(device)
            #batch_e = batch_graphs.edata['feat'].to(device)
            batch_e = None
            batch_labels = batch_labels.to(device)
            data_mask_c, data_mask_node_c = masker_c(batch_graphs, batch_x, batch_e)
            data_mask_b = 1-data_mask_c
            data_mask_node_b = 1-data_mask_node_c
            if binary:
                data_mask = pruning.binary_mask(data_mask, args.pa)
            #if epoch < 150:
            #    data_mask_c = None
            #    data_mask_b = None
            if not args.use_mask:
                data_mask_c = None
                data_mask_b = None
            batch_scores_c = model_c(batch_graphs, batch_x, batch_e, data_mask_c, data_mask_node_c)

            batch_scores_b = model_b(batch_graphs, batch_x, batch_e, data_mask_b, data_mask_node_b)
            scores_concat = torch.cat((batch_scores_c, batch_scores_b), dim=1)
            batch_scores = model_c.MLP_layer(scores_concat)
            loss = model_c.loss(batch_scores, batch_labels).mean() 
            epoch_test_loss += loss.detach().item()
            epoch_test_acc += accuracy(batch_scores, batch_labels)
            nb_data += batch_labels.size(0)
        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= nb_data
        
    return epoch_test_loss, epoch_test_acc



def train_epoch(model, optimizer, device, data_loader, epoch, args):

    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    gpu_mem = 0
    for iter, (batch_graphs, batch_labels, batch_bias_labels) in enumerate(data_loader):
        batch_graphs = dgl.batch(batch_graphs).to(device)
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        #batch_e = batch_graphs.edata['feat'].to(device)
        batch_e = None
        batch_labels = batch_labels.to(device)
        optimizer.zero_grad()
        
        batch_scores = model.forward(batch_graphs, batch_x, batch_e)
        loss = model.loss(batch_scores, batch_labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_train_acc += accuracy(batch_scores, batch_labels)
        nb_data += batch_labels.size(0)
    
        if iter % 40 == 0:
            print('-'*120)
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' | ' +
                    'Epoch: [{}/{}]  Iter: [{}/{}]  Loss: [{:.4f}]'
                    .format(epoch + 1, args.eval_epochs, iter, len(data_loader), epoch_loss / (iter + 1), epoch_train_acc / nb_data * 100))
    
    epoch_loss /= (iter + 1)
    epoch_train_acc /= nb_data

    return epoch_loss, epoch_train_acc, optimizer



def evaluate_network(model, device, data_loader, epoch):
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0
    with torch.no_grad():
        for iter, (batch_graphs, batch_labels, batch_bias_labels) in enumerate(data_loader):
            batch_graphs = dgl.batch(batch_graphs).to(device)
            batch_x = batch_graphs.ndata['feat'].to(device)
            #batch_e = batch_graphs.edata['feat'].to(device)
            batch_e = None
            batch_labels = batch_labels.to(device)
            
            batch_scores = model.forward(batch_graphs, batch_x, batch_e)
            loss = model.loss(batch_scores, batch_labels) 
            epoch_test_loss += loss.detach().item()
            epoch_test_acc += accuracy(batch_scores, batch_labels)
            nb_data += batch_labels.size(0)
        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= nb_data
        
    return epoch_test_loss, epoch_test_acc

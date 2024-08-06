#! /usr/bin/python
# -*- encoding: utf-8 -*-
# Adapted from https://github.com/CoinCheung/pytorch-loss (MIT License)

import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import accuracy
from pytorch_metric_learning import losses

class contrastive_amsoftmax(nn.Module):
    def __init__(self, embedding_dim, num_classes, loss_type='supcon', use_outputs=True, all_blocks=True, margin=0.2, scale=30, **kwargs):
        super(contrastive_amsoftmax, self).__init__()

        self.m = margin
        self.s = scale
        self.in_feats = embedding_dim
        self.W = torch.nn.Parameter(torch.randn(embedding_dim, num_classes), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        self.loss_type = loss_type
        self.use_outputs = use_outputs
        self.all_blocks = all_blocks
        nn.init.xavier_normal_(self.W, gain=1)

        print('Initialised AM-Softmax m=%.3f s=%.3f'%(self.m, self.s))
        print('Embedding dim is {}, number of speakers is {}'.format(embedding_dim, num_classes))

    def forward(self, x, outputs, label=None):
        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.in_feats

        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)
        # outputs_norm = torch.norm(outputs, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        # outputs_norm = torch.div(outputs, outputs_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.W, w_norm)
        costh = torch.mm(x_norm, w_norm)
        label_view = label.view(-1, 1)
        if label_view.is_cuda: label_view = label_view.cpu()
        delt_costh = torch.zeros(costh.size()).scatter_(1, label_view, self.m)
        if x.is_cuda: delt_costh = delt_costh.cuda()
        costh_m = costh - delt_costh
        costh_m_s = self.s * costh_m
        loss = self.ce(costh_m_s, label)
        acc = accuracy(costh_m_s.detach(), label.detach(), topk=(1,))[0]

        # Default
        # contrastive_loss = 0

        contrastive_loss = self.contrastive_loss_function(x, outputs, label) # loss_type='supcon', use_outputs=True, all_blocks=True
        final_loss = loss + contrastive_loss

        return final_loss, acc, contrastive_loss
    

    def contrastive_loss_function(self, x, outputs, labels): # loss_type='supcon', use_outputs=True, all_blocks=True
        loss_type = self.loss_type
        use_outputs = self.use_outputs
        all_blocks = self.all_blocks
        
        if use_outputs:
            outputs = outputs
        else:
            outputs = x

        if loss_type=='triplet':
            criterion = losses.TripletMarginLoss()
        elif loss_type=='ntxent':
            criterion = losses.NTXentLoss()
        elif loss_type=='supcon':
            criterion = losses.SupConLoss()

        outputs = outputs.squeeze()
        if all_blocks==True:
            embed_len = int(outputs.shape[1]/6)
            output_1 = outputs[:, :embed_len]
            output_2 = outputs[:, embed_len:2*embed_len]
            output_3 = outputs[:, 2*embed_len:3*embed_len]
            output_4 = outputs[:, 3*embed_len:4*embed_len]
            output_5 = outputs[:, 4*embed_len:5*embed_len]
            output_6 = outputs[:, 5*embed_len:6*embed_len]
            contrastive_loss = (criterion(output_1, labels) + criterion(output_2, labels) + criterion(output_3, labels) + criterion(output_4, labels) + criterion(output_5, labels) + criterion(output_6, labels))/6
        else:
            contrastive_loss = criterion(outputs, labels)

        if loss_type!='triplet': 
            contrastive_loss = 0.1*contrastive_loss
        else:
            contrastive_loss = 100*contrastive_loss

        return contrastive_loss
















    # def nt_bce_loss(self, embeddings, labels, temperature=0.5):
    #     """
    #     Compute the NT binary cross entropy loss.

    #     Args:
    #         embeddings (torch.Tensor): Batch of embeddings with shape (batch_size, embedding_dim).
    #         labels (torch.Tensor): Corresponding labels with shape (batch_size,).
    #         temperature (float): Temperature scaling factor. Default is 0.5.

    #     Returns:
    #         torch.Tensor: Computed NT binary cross entropy loss.
    #     """
    #     batch_size = embeddings.size()[0]
    #     embedding_dim = embeddings.size()[1]
        
    #     # Normalize embeddings
    #     embeddings = F.normalize(embeddings, p=2, dim=1)
        
    #     # Compute cosine similarity
    #     similarity_matrix = torch.matmul(embeddings, embeddings.T) / temperature
        
    #     # Create masks for positive and negative pairs
    #     labels = labels.view(batch_size, 1)
    #     mask_positive = (labels == labels.T).float()
    #     mask_negative = (labels != labels.T).float()
        
    #     # Labels for binary cross-entropy
    #     target_labels = mask_positive.clone()
        
    #     # Apply sigmoid to similarity matrix
    #     similarity_matrix = torch.sigmoid(similarity_matrix)
        
    #     # Compute binary cross-entropy loss
    #     bce_loss = F.binary_cross_entropy(similarity_matrix, target_labels, reduction='sum')
        
    #     # Normalize by number of elements
    #     loss = bce_loss / (batch_size * batch_size)
        
    #     return loss


    # def triplet_loss(self, embeddings, labels, margin=1.0):
    #     """
    #     Compute the triplet loss.

    #     Args:
    #         embeddings (torch.Tensor): Batch of embeddings with shape (batch_size, embedding_dim).
    #         labels (torch.Tensor): Corresponding labels with shape (batch_size,).
    #         margin (float): Margin for the triplet loss. Default is 1.0.

    #     Returns:
    #         torch.Tensor: Computed triplet loss.
    #     """
    #     batch_size = embeddings.size()[0]
    #     # embedding_dim = embeddings.size()[1]
        
    #     # Create pairwise distance matrix
    #     distances = torch.cdist(embeddings, embeddings, p=2)

    #     # Create masks for anchor-positive and anchor-negative pairs
    #     labels = labels.view(batch_size, 1)
    #     mask_positive = (labels == labels.t()).float()
    #     mask_negative = (labels != labels.t()).float()

    #     # Select triplets
    #     triplet_loss = 0.0
    #     num_triplets = 0

    #     for i in range(batch_size):
    #         for j in range(batch_size):
    #             if i != j and mask_positive[i, j] > 0:
    #                 for k in range(batch_size):
    #                     if i != k and j != k and mask_negative[i, k] > 0:
    #                         # Compute the triplet loss for the triplet (i, j, k)
    #                         dist_anchor_positive = distances[i, j]
    #                         dist_anchor_negative = distances[i, k]
    #                         loss = F.relu(dist_anchor_positive - dist_anchor_negative + margin)
    #                         triplet_loss += loss
    #                         num_triplets += 1

    #     # Average the triplet loss over all valid triplets
    #     if num_triplets > 0:
    #         triplet_loss /= num_triplets

    #     return triplet_loss


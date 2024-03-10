# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import math
from .layers import MultiHeadGraphAttention, GraphConvolution

class GAT(nn.Module):
    def __init__(self, n_units, n_heads, dropout, attn_dropout, instance_normalization, diag):
        super(GAT, self).__init__()
        self.num_layer = len(n_units) - 1
        self.dropout = dropout
        self.inst_norm = instance_normalization
        if self.inst_norm:
            self.norm = nn.InstanceNorm1d(n_units[0], momentum=0.0, affine=True)
        self.layer_stack = nn.ModuleList()
        self.diag = diag
        for i in range(self.num_layer):
            f_in = n_units[i] * n_heads[i - 1] if i else n_units[i]
            self.layer_stack.append(MultiHeadGraphAttention(n_heads[i], f_in, n_units[i + 1], attn_dropout, diag, nn.init.ones_, False))

    def forward(self, x, adj):
        if self.inst_norm:
            x = self.norm(x)
        for i, gat_layer in enumerate(self.layer_stack):
            if i + 1 < self.num_layer:
                x = F.dropout(x, self.dropout, training=self.training)
            x = gat_layer(x, adj)
            if self.diag:
                x = x.mean(dim=0)
            if i + 1 < self.num_layer:
                if self.diag:
                    x = F.elu(x)
                else:
                    x = F.elu(x.transpose(0, 1).contiguous().view(adj.size(0), -1))
        if not self.diag:
            x = x.mean(dim=0)

        return x

""" vanilla GCN """

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nout)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))  # change to leaky relu
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x

""" loss """

def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())

def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    a = norm.expand_as(X) + 1e-8
    X = torch.div(X, a)
    return X

class NCA_loss(nn.Module):

    def __init__(self, alpha, beta, ep):
        super(NCA_loss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.ep = ep
        self.sim = cosine_sim
        # from pytorch_metric_learning import losses
        # self.loss_func = losses.MultiSimilarityLoss()

    def forward(self, emb, train_links, test_links):

        emb = F.normalize(emb)
        num_ent = emb.shape[0]

        im = emb[train_links[:, 0]]
        s = emb[train_links[:, 1]]

        # labels = torch.arange(im.size(0))
        # embeddings = torch.cat([im, s], dim=0)
        # labels = torch.cat([labels, labels], dim=0)
        # loss = self.loss_func(embeddings, labels)
        # return loss

        # """

        if len(test_links) != 0:
            test_links = test_links[random.sample([x for x in np.arange(0, len(test_links))], 4500)]

            im_neg_scores = self.sim(im, emb[test_links[:, 1]])
            s_neg_scores = self.sim(s, emb[test_links[:, 0]])

        # im = l2norm(im)
        # s = l2norm(s)

        bsize = im.size()[0]
        # compute prediction-target score matrix
        # print (im)
        # print(s)
        scores = self.sim(im, s)  # + 1
        # print (scores)
        tmp = torch.eye(bsize).cuda()
        s_diag = tmp * scores

        alpha = self.alpha
        alpha_2 = alpha  # / 3.0
        beta = self.beta
        ep = self.ep
        S_ = torch.exp(alpha * (scores - ep))
        S_ = S_ - S_ * tmp  # clear diagnal

        if len(test_links) != 0:
            S_1 = torch.exp(alpha * (im_neg_scores - ep))
            S_2 = torch.exp(alpha * (s_neg_scores - ep))

        loss_diag = - torch.log(1 + F.relu(s_diag.sum(0)))

        
        
        loss = torch.sum(
            torch.log(1 + S_.sum(0)) / alpha + torch.log(1 + S_.sum(1)) / alpha + loss_diag * beta) / bsize
        if len(test_links) != 0:
            loss_global_neg = (torch.sum(torch.log(1 + S_1.sum(0)) / alpha_2 + torch.log(1 + S_2.sum(0)) / alpha_2) +
                               torch.sum(torch.log(1 + S_1.sum(1)) / alpha_2 + torch.log(1 + S_2.sum(1)) / alpha_2)) / 4500
        if len(test_links) != 0:
            return loss + loss_global_neg
        return loss
        # """

class NCA_loss_cross_modal(nn.Module):

    def __init__(self, alpha, beta, ep):
        super(NCA_loss_cross_modal, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.ep = ep
        self.sim = cosine_sim

    def forward(self, emb1, emb2, train_links):

        emb1 = F.normalize(emb1)
        emb2 = F.normalize(emb2)
        num_ent = emb1.shape[0]

        im = emb1[train_links[:, 0]]
        s = emb2[train_links[:, 1]]

        bsize = im.size()[0]
        # compute prediction-target score matrix
        # print (im)
        # print(s)
        scores = self.sim(im, s)  # + 1
        # print (scores)
        tmp = torch.eye(bsize).cuda()
        s_diag = tmp * scores

        alpha = self.alpha
        alpha_2 = alpha  # / 3.0
        beta = self.beta
        ep = self.ep
        S_ = torch.exp(alpha * (scores - ep))
        S_ = S_ - S_ * tmp  # clear diagnal

        loss_diag = - torch.log(1 + F.relu(s_diag.sum(0)))

        loss = torch.sum(
            torch.log(1 + S_.sum(0)) / alpha + torch.log(1 + S_.sum(1)) / alpha + loss_diag * beta) / bsize
        return loss

class SelfAttention(nn.Module):
    def __init__(self, hidden_size, batch_first=False):
        super(SelfAttention, self).__init__()

        self.hidden_size = hidden_size
        self.batch_first = batch_first

        self.att_weights = nn.Parameter(torch.Tensor(1, hidden_size),
                                        requires_grad=True)

        nn.init.xavier_uniform(self.att_weights.data)

    def get_mask(self):
        pass

    def forward(self, inputs):

        if self.batch_first:
            batch_size, max_len = inputs.size()[:2]
        else:
            max_len, batch_size = inputs.size()[:2]
            inputs = inputs.permute(1, 0, 2)

        # att = torch.mul(inputs, self.att_weights.expand_as(inputs))
        # att = att.sum(-1)
        weights = torch.bmm(inputs,
                            self.att_weights  # (1, hidden_size)
                            .permute(1, 0)  # (hidden_size, 1)
                            .unsqueeze(0)  # (1, hidden_size, 1)
                            # (batch_size, hidden_size, 1)
                            .repeat(batch_size, 1, 1)
                            )

        attentions = F.softmax(F.relu(weights.squeeze()))

        # apply weights
        weighted = torch.mul(
            inputs, attentions.unsqueeze(-1).expand_as(inputs))

        # get the final fixed vector representations of the sentences
        representations = weighted.sum(1).squeeze()

        return representations, attentions

def attention(q, k, v, d_k, mask=None, dropout=None):

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output

class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
            / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.0):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        # print (self.d_k)
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):

        bs = q.size(0)

        # perform linear operation and split into h heads

        # print (k.shape)

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)  # calculate attention using function we will define next
        # print (k.shape)
        scores = attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous()\
            .view(bs, -1, self.d_model)

        output = concat
        output = self.out(concat)

        return output

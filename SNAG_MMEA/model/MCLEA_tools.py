
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

from .layers import ProjectionHead
from .EVA_tools import GAT, GCN


class MultiModalFusion(nn.Module):
    def __init__(self, modal_num, with_weight=1):
        super().__init__()
        self.modal_num = modal_num
        self.requires_grad = True if with_weight > 0 else False
        
        self.weight = nn.Parameter(torch.ones((self.modal_num, 1)),
                                   requires_grad=self.requires_grad)

    def forward(self, embs):
        # assert len(embs) == self.modal_num
        weight_norm = F.softmax(self.weight, dim=0)
        embs = [weight_norm[idx] * F.normalize(embs[idx]) for idx in range(self.modal_num) if embs[idx] is not None]
        # joint_emb = torch.cat(embs, dim=1)

        hidden_states = torch.stack(embs, dim=1)
        joint_emb = hidden_states.mean(dim=1)
        # joint_emb = torch.sum(torch.stack(embs, dim=1), dim=1)
        return joint_emb

class MultiModalEncoder(nn.Module):
    """
    entity embedding: (ent_num, input_dim)
    gcn layer: n_units

    """

    def __init__(self, args,
                 ent_num,
                 img_feature_dim,
                 char_feature_dim=None,
                 use_project_head=False,
                 attr_input_dim=1000):
        super(MultiModalEncoder, self).__init__()

        self.args = args
        attr_dim = self.args.attr_dim
        img_dim = self.args.img_dim
        name_dim = self.args.name_dim
        char_dim = self.args.char_dim
        dropout = self.args.dropout
        self.ENT_NUM = ent_num
        self.use_project_head = use_project_head

        self.n_units = [int(x) for x in self.args.hidden_units.strip().split(",")]
        self.n_heads = [int(x) for x in self.args.heads.strip().split(",")]
        self.input_dim = int(self.args.hidden_units.strip().split(",")[0])

        #########################
        # Entity Embedding
        #########################
        self.entity_emb = nn.Embedding(self.ENT_NUM, self.input_dim)
        nn.init.normal_(self.entity_emb.weight, std=1.0 / math.sqrt(self.ENT_NUM))
        self.entity_emb.requires_grad = True

        #########################
        # Modal Encoder
        #########################

        self.rel_fc = nn.Linear(1000, attr_dim)
        self.att_fc = nn.Linear(attr_input_dim, attr_dim)
        self.img_fc = nn.Linear(img_feature_dim, img_dim)
        self.name_fc = nn.Linear(300, char_dim)
        self.char_fc = nn.Linear(char_feature_dim, char_dim)

        # structure encoder
        if self.args.structure_encoder == "gcn":
            self.cross_graph_model = GCN(self.n_units[0], self.n_units[1], self.n_units[2],
                                         dropout=self.args.dropout)
        elif self.args.structure_encoder == "gat":
            self.cross_graph_model = GAT(n_units=self.n_units, n_heads=self.n_heads, dropout=args.dropout,
                                         attn_dropout=args.attn_dropout,
                                         instance_normalization=self.args.instance_normalization, diag=True)

        #########################
        
        #########################
        if self.use_project_head:
            self.img_pro = ProjectionHead(img_dim, img_dim, img_dim, dropout)
            self.att_pro = ProjectionHead(attr_dim, attr_dim, attr_dim, dropout)
            self.rel_pro = ProjectionHead(attr_dim, attr_dim, attr_dim, dropout)
            self.gph_pro = ProjectionHead(self.n_units[2], self.n_units[2], self.n_units[2], dropout)

        #########################
        # Fusion Encoder
        #########################
        
        self.fusion = MultiModalFusion(modal_num=self.args.inner_view_num,
                                       with_weight=self.args.with_weight)

    def _emb_generate(self, input_idx, adj, img_features, rel_features, att_features, name_features=None, char_features=None, entity_noise=None,entity_noise_mask=None):
        if self.args.w_gcn:
            if entity_noise is None and entity_noise_mask is None:
                gph_emb = self.cross_graph_model(self.entity_emb(input_idx), adj)
            else:
                # pdb.set_trace()
                entity_emb = self.entity_emb(input_idx)
                entity_emb[entity_noise_mask] = (1.0 - self.args.mask_ratio * 0.5) * entity_emb[entity_noise_mask] + self.args.mask_ratio * 0.5 * entity_noise[entity_noise_mask]
                gph_emb = self.cross_graph_model(entity_emb, adj)
        else:
            gph_emb = None
        if self.args.w_img:
            img_emb = self.img_fc(img_features)
        else:
            img_emb = None
        if self.args.w_rel:
            rel_emb = self.rel_fc(rel_features)
        else:
            rel_emb = None
        if self.args.w_attr:
            att_emb = self.att_fc(att_features)
        else:
            att_emb = None
        if self.args.w_name:
            name_emb = self.name_fc(name_features)
        else:
            name_emb = None
        if self.args.w_char:
            char_emb = self.char_fc(char_features)
        else:
            char_emb = None

        return (gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb)

    def forward(self,
                gph_emb=None,
                img_emb=None,
                rel_emb=None,
                att_emb=None,
                name_emb=None,
                char_emb=None):

        if self.use_project_head:
            gph_emb = self.gph_pro(gph_emb)
            img_emb = self.img_pro(img_emb)
            rel_emb = self.rel_pro(rel_emb)
            att_emb = self.att_pro(att_emb)

        joint_emb = self.fusion([img_emb, att_emb, rel_emb, gph_emb, name_emb, char_emb])

        return joint_emb

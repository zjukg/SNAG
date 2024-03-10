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
import pdb

class AttrEncoder(nn.Module):
    def __init__(self, kgs, args):
        super().__init__()
        self.args = args
        
        self.attr_embed = nn.Embedding.from_pretrained(torch.FloatTensor(kgs["att_features"]))
        self.fc1 = nn.Linear(kgs["att_features"].shape[1], self.args.dim)
        self.fc2 = nn.Linear(self.args.dim, self.args.dim)
        nn.init.xavier_normal_(self.fc1.weight.data)
        nn.init.xavier_normal_(self.fc2.weight.data)

    def forward(self, e_idx, e_i):
        # pdb.set_trace()
        e_a = self.fc1(self.attr_embed(e_idx))
        # e_v = torch.sigmoid(e_v.unsqueeze(-1)).repeat(1, 1, self.args.dim)
        # e = self.fc2(torch.cat([e_a, e_v], dim=2))
        # Vision-adaptive Attribute Learning
        # alpha = F.softmax(torch.sum(e_a * e_i.unsqueeze(1), dim=-1), dim=1)
        # e = torch.sum(alpha.unsqueeze(2) * e_a, dim=1)
        return e_a

def generate_relation_triple_batch(triple_list1, triple_list2, triple_set1, triple_set2,
                                   entity_list1, entity_list2, batch_size, step, neg_triples_num):
    # def generate_relation_triple_batch(batch, triple_set, entity_list, neg_triples_num):
    batch_size1 = int(len(triple_list1) / (len(triple_list1) + len(triple_list2)) * batch_size)
    batch_size2 = batch_size - batch_size1
    pos_batch1 = generate_pos_triples(triple_list1, batch_size1, step)
    pos_batch2 = generate_pos_triples(triple_list2, batch_size2, step)

    neg_batch1 = generate_neg_triples_fast(pos_batch1, triple_set1, entity_list1, neg_triples_num)
    neg_batch2 = generate_neg_triples_fast(pos_batch2, triple_set2, entity_list2, neg_triples_num)
    return pos_batch1 + pos_batch2, neg_batch1 + neg_batch2

def generate_pos_triples(triples, batch_size, step, is_fixed_size=False):
    start = (step * batch_size) % len(triples)
    end = start + batch_size
    if end > len(triples):
        end = len(triples)
    pos_batch = triples[start: end]
    if is_fixed_size and len(pos_batch) < batch_size:
        pos_batch += triples[:batch_size - len(pos_batch)]
    return pos_batch

def generate_neg_triples_fast(pos_batch, all_triples_set, entities_list, neg_triples_num, neighbor=None, max_try=10):
    if neighbor is None:
        neighbor = dict()
    neg_batch = list()
    for head, relation, tail in pos_batch:
        neg_triples = list()
        nums_to_sample = neg_triples_num
        head_candidates = neighbor.get(head, entities_list)
        tail_candidates = neighbor.get(tail, entities_list)
        for i in range(max_try):
            corrupt_head_prob = np.random.binomial(1, 0.5)
            if corrupt_head_prob:
                neg_heads = random.sample(head_candidates, nums_to_sample)
                i_neg_triples = {(h2, relation, tail) for h2 in neg_heads}
            else:
                neg_tails = random.sample(tail_candidates, nums_to_sample)
                i_neg_triples = {(head, relation, t2) for t2 in neg_tails}
            if i == max_try - 1:
                neg_triples += list(i_neg_triples)
                break
            else:
                
                i_neg_triples = list(i_neg_triples - all_triples_set)
                neg_triples += i_neg_triples
            if len(neg_triples) == neg_triples_num:
                break
            else:
                nums_to_sample = neg_triples_num - len(neg_triples)
        assert len(neg_triples) == neg_triples_num
        neg_batch.extend(neg_triples)
    assert len(neg_batch) == neg_triples_num * len(pos_batch)
    return neg_batch

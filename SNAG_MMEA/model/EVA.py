
import types
import torch
import transformers
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import numpy as np
import pdb
import math
from .Tool_model import AutomaticWeightedLoss
from .EVA_tools import GCN, NCA_loss
from src.utils import pairwise_distances
import os.path as osp
import json

class EVA(nn.Module):
    def __init__(self, kgs, args):
        super().__init__()
        self.ent_num = kgs["ent_num"]
        self.rel_num = kgs["rel_num"]
        self.kgs = kgs
        self.args = args
        
        
        
        self.input_dim = int(args.hidden_units.strip().split(",")[0])
        n_units = [int(x) for x in args.hidden_units.strip().split(",")]

        param_num = self.args.inner_view_num
        params = torch.ones(param_num, requires_grad=True)
        self.weight_raw = torch.nn.Parameter(params)

        # self.weight_raw = torch.tensor([1.0, 1.0, 1.0, 1.0], requires_grad=True).cuda()
        self.img_embed = F.normalize(torch.FloatTensor(kgs["images_list"])).cuda()
        self.rel_embed = torch.Tensor(kgs["rel_features"]).cuda()
        self.attr_embed = torch.Tensor(kgs["att_features"]).cuda()
        
        self.ent_embed = nn.Embedding(self.ent_num, self.input_dim)
        self.ent_wo_img = torch.tensor(kgs['ent_wo_img']).cuda()
        attr_dim = self.args.attr_dim
        
        self.rel_fc = nn.Linear(1000, attr_dim)
        
        self.att_fc = nn.Linear(kgs["att_features"].shape[1], attr_dim)
        
        self.img_fc = nn.Linear(self._get_img_dim(kgs), attr_dim)
        self.ent_embed.requires_grad = True
        self.input_idx = kgs["input_idx"].cuda()
        self.adj = kgs["adj"].cuda()
        # "400,400,200"
        self.cross_graph_model = GCN(n_units[0], n_units[1], n_units[2], dropout=args.dropout)

        
        nn.init.xavier_normal_(self.rel_fc.weight.data)
        nn.init.xavier_normal_(self.att_fc.weight.data)
        nn.init.xavier_normal_(self.img_fc.weight.data)
        nn.init.xavier_normal_(self.ent_embed.weight.data)

        self.criterion_gcn = NCA_loss(alpha=5, beta=10, ep=0.0)
        self.criterion_rel = NCA_loss(alpha=15, beta=10, ep=0.0)
        self.criterion_att = NCA_loss(alpha=15, beta=10, ep=0.0)
        self.criterion_img = NCA_loss(alpha=15, beta=10, ep=0.0)

        # name and attr
        if args.w_name and args.w_char:
            self.name_features = kgs["name_features"].cuda()
            self.char_features = kgs["char_features"].cuda()
            char_feature_dim = kgs["char_features"].shape[1] if self.char_features is not None else 100
            self.char_fc = nn.Linear(char_feature_dim, args.char_dim)
            self.name_fc = nn.Linear(300, args.char_dim)
            nn.init.xavier_normal_(self.char_fc.weight.data)
            nn.init.xavier_normal_(self.name_fc.weight.data)
            self.criterion_name = NCA_loss(alpha=15, beta=10, ep=0.0)
            self.criterion_char = NCA_loss(alpha=15, beta=10, ep=0.0)

        # joint loss
        self.criterion_all = NCA_loss(alpha=15, beta=10, ep=0.0)

        if self.args.add_noise:
            self.get_mean_std()

    def add_noise_to_embeddings(self, embeddings, mean, std, noise_ratio=0.1):
        
        noise_mask = torch.rand(embeddings.shape[0]) < noise_ratio
        selected_embeddings = embeddings[noise_mask]
        noise = mean + std * torch.randn_like(selected_embeddings)
        # pdb.set_trace()
        
        # embeddings[noise_mask] = 0.95 * selected_embeddings + 0.05 * noise
        embeddings[noise_mask] = (1.0 - self.args.mask_ratio) * selected_embeddings + self.args.mask_ratio * noise
        return embeddings

    def get_mean_std(self):
        valid_img_emb = torch.cat([self.img_embed[i].unsqueeze(0) for i in range(self.img_embed.size(0)) if i not in self.ent_wo_img], dim=0)
        self.img_mean = torch.mean(valid_img_emb, dim=0).cuda()
        self.img_std = torch.std(valid_img_emb, dim=0).cuda()
        self.rel_mean = torch.mean(self.rel_embed, dim=0).cuda()
        self.rel_std = torch.std(self.rel_embed, dim=0).cuda()
        self.att_mean = torch.mean(self.attr_embed, dim=0).cuda()
        self.att_std = torch.std(self.attr_embed, dim=0).cuda()

    def update_noise(self):
        
        
        self.rel_noisy_features = self.add_noise_to_embeddings(self.rel_embed.clone(), self.rel_mean, self.rel_std, noise_ratio=self.args.noise_ratio)
        self.att_noisy_features = self.add_noise_to_embeddings(self.attr_embed.clone(), self.att_mean, self.att_std, noise_ratio=self.args.noise_ratio)
        self.img_noisy_features = self.add_noise_to_embeddings(self.img_embed.clone(), self.img_mean, self.img_std, noise_ratio=self.args.noise_ratio)
        
        
        self.ent_mean = torch.mean(self.ent_embed.weight.data, dim=0)
        self.ent_std = torch.std(self.ent_embed.weight.data, dim=0)
        self.entity_noise = self.ent_mean + self.ent_std * torch.randn_like(self.ent_embed.weight.data)
        # self.entity_noise_mask = torch.rand(self.ent_embeddings.weight.shape[0]) < self.args.noise_ratio
        self.entity_noise_mask = (torch.rand(self.ent_embed.weight.shape[0]) < self.args.noise_ratio * 0.5).cuda()
        # pdb.set_trace()

    def _get_img_dim(self, kgs):
        if isinstance(kgs["images_list"], list):
            img_dim = kgs["images_list"][0].shape[1]
        elif isinstance(kgs["images_list"], np.ndarray) or torch.is_tensor(kgs["images_list"]):
            img_dim = kgs["images_list"].shape[1]
        return img_dim

    def emb_generat(self):
        # pdb.set_trace()
        if self.args.add_noise and self.img_fc.training:
            entity_emb = self.ent_embed(self.input_idx)
            entity_emb[self.entity_noise_mask] = (1.0 - self.args.mask_ratio * 0.5) * entity_emb[self.entity_noise_mask] + self.args.mask_ratio * 0.5 * self.entity_noise[self.entity_noise_mask]
            self.gph_emb = self.cross_graph_model(entity_emb, self.adj)
            self.img_emb = self.img_fc(self.img_noisy_features)
            self.rel_emb = self.rel_fc(self.rel_noisy_features)
            self.att_emb = self.att_fc(self.att_noisy_features)
            # return gph_emb, img_emb, rel_emb, att_emb
        else:
            self.gph_emb = self.cross_graph_model(self.ent_embed(self.input_idx), self.adj)
            self.img_emb = self.img_fc(self.img_embed)
            self.rel_emb = self.rel_fc(self.rel_embed)
            self.att_emb = self.att_fc(self.attr_embed)
            # return gph_emb, img_emb, rel_emb, att_emb

        if self.args.w_name and self.args.w_char:
            self.name_emb = self.name_fc(self.name_features)
            self.char_emb = self.char_fc(self.char_features)

    def joint_emb_generat(self):
        w_normalized = F.softmax(self.weight_raw, dim=0)
        # ablation
        if self.args.w_name and self.args.w_char:
            joint_emb = torch.cat([
                w_normalized[0] * F.normalize(self.img_emb).detach(),
                w_normalized[1] * F.normalize(self.att_emb).detach(),
                w_normalized[2] * F.normalize(self.rel_emb).detach(),
                w_normalized[3] * F.normalize(self.gph_emb).detach(),
                w_normalized[4] * F.normalize(self.name_emb).detach(),
                w_normalized[5] * F.normalize(self.char_emb).detach(),
            ], dim=1)
        else:
            joint_emb = torch.cat([
                w_normalized[0] * F.normalize(self.img_emb).detach(),
                w_normalized[1] * F.normalize(self.att_emb).detach(),
                w_normalized[2] * F.normalize(self.rel_emb).detach(),
                w_normalized[3] * F.normalize(self.gph_emb).detach(),
            ], dim=1)
        return joint_emb

    def forward(self, batch, cache=None):
        self.emb_generat()
        loss_sum_gcn, loss_sum_rel, loss_sum_att, loss_sum_img = 0, 0, 0, 0
        joint_emb = self.joint_emb_generat()
        loss_GCN = self.criterion_gcn(self.gph_emb, batch, [])
        loss_rel = self.criterion_rel(self.rel_emb, batch, [])
        loss_att = self.criterion_att(self.att_emb, batch, [])
        loss_img = self.criterion_img(self.img_emb, batch, [])
        loss_joi = self.criterion_all(joint_emb, batch, [])

        if self.args.w_name and self.args.w_char:
            loss_name = self.criterion_name(self.name_emb, batch, [])
            loss_char = self.criterion_char(self.char_emb, batch, [])
            loss_all = loss_joi + loss_att + loss_rel + loss_GCN + loss_img + loss_name + loss_char
            loss_dic = {"gcn": loss_GCN.item(), "rel": loss_rel.item(), "att": loss_att.item(), "img": loss_img.item(), "joi": loss_joi.item(), "name": loss_name.item(), "char": loss_char.item()}
        else:
            # ablation
            loss_all = loss_joi + loss_att + loss_rel + loss_GCN + loss_img
            loss_dic = {"gcn": loss_GCN.item(), "rel": loss_rel.item(), "att": loss_att.item(), "img": loss_img.item(), "joi": loss_joi.item()}
        # loss_all = loss_joi + loss_GCN + loss_img
        output = {"loss_dic": loss_dic, "emb": joint_emb, "weight": self.weight_raw.tolist()}
        return loss_all, output

    def Iter_new_links(self, epoch, left_non_train, final_emb, right_non_train, new_links=[]):
        if len(left_non_train) == 0 or len(right_non_train) == 0:
            return new_links
        distance_list = []
        for i in np.arange(0, len(left_non_train), 1000):
            d = pairwise_distances(final_emb[left_non_train[i:i + 1000]], final_emb[right_non_train])
            distance_list.append(d)
        distance = torch.cat(distance_list, dim=0)
        preds_l = torch.argmin(distance, dim=1).cpu().numpy().tolist()
        preds_r = torch.argmin(distance.t(), dim=1).cpu().numpy().tolist()
        del distance_list, distance, final_emb
        if (epoch + 1) % (self.args.semi_learn_step * 10) == self.args.semi_learn_step:
            
            new_links = [(left_non_train[i], right_non_train[p]) for i, p in enumerate(preds_l) if preds_r[p] == i]
            
        else:
            new_links = [(left_non_train[i], right_non_train[p]) for i, p in enumerate(preds_l) if (preds_r[p] == i) and ((left_non_train[i], right_non_train[p]) in new_links)]

        # if self.args.rank == 0:
        #     print("[epoch %d] #links in candidate set: %d" % (epoch, len(new_links)))
        return new_links

    def data_refresh(self, logger, train_ill, test_ill_, left_non_train, right_non_train, new_links=[]):
        if len(new_links) != 0 and (len(left_non_train) != 0 and len(right_non_train) != 0):
            new_links_select = new_links
            # if len(new_links) >= 5000: new_links = random.sample(new_links, 5000)
            train_ill = np.vstack((train_ill, np.array(new_links_select)))
            num_true = len([nl for nl in new_links_select if nl in test_ill_])
            # remove from left/right_non_train
            for nl in new_links_select:
                left_non_train.remove(nl[0])
                right_non_train.remove(nl[1])

            # if self.args.rank == 0:
            logger.info(f"#new_links_select:{len(new_links_select)}")
            logger.info(f"train_ill.shape:{train_ill.shape}")
            logger.info(f"#true_links: {num_true}")
            logger.info(f"true link ratio: {(100 * num_true / len(new_links_select)):.1f}%")
            logger.info(f"#entity not in train set: {len(left_non_train)} (left) {len(right_non_train)} (right)")

            new_links = []
        else:
            logger.info("len(new_links) is 0")

        return left_non_train, right_non_train, train_ill, new_links

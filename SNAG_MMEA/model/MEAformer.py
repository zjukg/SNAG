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
# from .MCLEA import MultiModalEncoder
from .MEAformer_tools import MultiModalEncoder
from .MEAformer_loss import CustomMultiLossLayer, ial_loss, icl_loss

from src.utils import pairwise_distances
import os.path as osp
import json

class MEAformer(nn.Module):
    def __init__(self, kgs, args):
        super().__init__()
        self.kgs = kgs
        self.args = args
        self.img_features = F.normalize(torch.FloatTensor(kgs["images_list"])).cuda()
        self.input_idx = kgs["input_idx"].cuda()
        self.adj = kgs["adj"].cuda()
        self.rel_features = torch.Tensor(kgs["rel_features"]).cuda()
        self.att_features = torch.Tensor(kgs["att_features"]).cuda()
        self.ent_wo_img = torch.tensor(kgs['ent_wo_img']).cuda()
        self.name_features = None
        self.char_features = None
        if kgs["name_features"] is not None:
            self.name_features = kgs["name_features"].cuda()
            self.char_features = kgs["char_features"].cuda()

        img_dim = self._get_img_dim(kgs)

        char_dim = kgs["char_features"].shape[1] if self.char_features is not None else 100

        
        
        self.multimodal_encoder = MultiModalEncoder(args=self.args,
                                                    ent_num=kgs["ent_num"],
                                                    img_feature_dim=img_dim,
                                                    char_feature_dim=char_dim,
                                                    use_project_head=self.args.use_project_head,
                                                    attr_input_dim=kgs["att_features"].shape[1])

        self.multi_loss_layer = CustomMultiLossLayer(loss_num=6)  # 6
        self.criterion_cl = icl_loss(tau=self.args.tau, ab_weight=self.args.ab_weight, n_view=2)
        self.criterion_cl_joint = icl_loss(tau=self.args.tau, ab_weight=self.args.ab_weight, n_view=2, replay=self.args.replay, neg_cross_kg=self.args.neg_cross_kg)
        # self.align_multi_loss_layer = CustomMultiLossLayer(loss_num=6)  # 6
        # self.criterion_align = ial_loss(tau=self.args.tau2, ab_weight=self.args.ab_weight, zoom=self.args.zoom, reduction=self.args.reduction)

        tmp = -1 * torch.ones(self.input_idx.shape[0], dtype=torch.int64).cuda()
        self.replay_matrix = torch.stack([self.input_idx, tmp], dim=1).cuda()
        
        self.replay_ready = 0
        self.idx_one = torch.ones(self.args.batch_size, dtype=torch.int64).cuda()
        self.idx_double = torch.cat([self.idx_one, self.idx_one]).cuda()
        self.last_num = 1000000000000
        # self.idx_one = np.ones(self.args.batch_size, dtype=np.int64)
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
        valid_img_emb = torch.cat([self.img_features[i].unsqueeze(0) for i in range(self.img_features.size(0)) if i not in self.ent_wo_img], dim=0)
        self.img_mean = torch.mean(valid_img_emb, dim=0).cuda()
        self.img_std = torch.std(valid_img_emb, dim=0).cuda()
        self.rel_mean = torch.mean(self.rel_features, dim=0).cuda()
        self.rel_std = torch.std(self.rel_features, dim=0).cuda()
        self.att_mean = torch.mean(self.att_features, dim=0).cuda()
        self.att_std = torch.std(self.att_features, dim=0).cuda()

    def update_noise(self):
        
        
        self.rel_noisy_features = self.add_noise_to_embeddings(self.rel_features.clone(), self.rel_mean, self.rel_std, noise_ratio=self.args.noise_ratio)
        self.att_noisy_features = self.add_noise_to_embeddings(self.att_features.clone(), self.att_mean, self.att_std, noise_ratio=self.args.noise_ratio)
        self.img_noisy_features = self.add_noise_to_embeddings(self.img_features.clone(), self.img_mean, self.img_std, noise_ratio=self.args.noise_ratio)
        
        
        self.ent_mean = torch.mean(self.multimodal_encoder.entity_emb.weight.data, dim=0)
        self.ent_std = torch.std(self.multimodal_encoder.entity_emb.weight.data, dim=0)
        self.entity_noise = self.ent_mean + self.ent_std * torch.randn_like(self.multimodal_encoder.entity_emb.weight.data)
        # self.entity_noise_mask = torch.rand(self.ent_embeddings.weight.shape[0]) < self.args.noise_ratio
        self.entity_noise_mask = (torch.rand(self.multimodal_encoder.entity_emb.weight.shape[0]) < self.args.noise_ratio * 0.5).cuda()
        # pdb.set_trace()

    def forward(self, batch):
        gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb, joint_emb, hidden_states = self.joint_emb_generat(only_joint=False)
        gph_emb_hid, rel_emb_hid, att_emb_hid, img_emb_hid, name_emb_hid, char_emb_hid, joint_emb_hid = self.generate_hidden_emb(hidden_states)
        
        
        
        if self.args.replay:
            batch = torch.tensor(batch, dtype=torch.int64).cuda()
            all_ent_batch = torch.cat([batch[:, 0], batch[:, 1]])
            if not self.replay_ready:
                loss_joi, l_neg, r_neg = self.criterion_cl_joint(joint_emb, batch)
            else:
                neg_l = self.replay_matrix[batch[:, 0], self.idx_one[:batch.shape[0]]]
                neg_r = self.replay_matrix[batch[:, 1], self.idx_one[:batch.shape[0]]]
                neg_l_set = set(neg_l.tolist())
                neg_r_set = set(neg_r.tolist())
                all_ent_set = set(all_ent_batch.tolist())
                neg_l_list = list(neg_l_set - all_ent_set)
                neg_r_list = list(neg_r_set - all_ent_set)
                neg_l_ipt = torch.tensor(neg_l_list, dtype=torch.int64).cuda()
                neg_r_ipt = torch.tensor(neg_r_list, dtype=torch.int64).cuda()
                
                
                # pdb.set_trace()
                loss_joi, l_neg, r_neg = self.criterion_cl_joint(joint_emb, batch, neg_l_ipt, neg_r_ipt)

            
            index = (
                all_ent_batch,
                self.idx_double[:batch.shape[0] * 2],
                # torch.cat([self.idx_one[:batch.shape[0]], self.idx_one[:batch.shape[0]]]),
            )
            new_value = torch.cat([l_neg, r_neg]).cuda()

            self.replay_matrix = self.replay_matrix.index_put(index, new_value)
            
            if self.replay_ready == 0:
                
                num = torch.sum(self.replay_matrix < 0)
                if num == self.last_num:
                    
                    self.replay_ready = 1
                    print("-----------------------------------------")
                    print("begin replay!")
                    print("-----------------------------------------")
                else:
                    self.last_num = num
        else:
            loss_joi = self.criterion_cl_joint(joint_emb, batch)
        
        # loss_joi_hid = self.criterion_cl(joint_emb_hid, batch)

        
        # ICL loss for uni-modal embedding
        in_loss = self.inner_view_loss(gph_emb, rel_emb, att_emb, img_emb, name_emb, char_emb, batch)
        
        out_loss = self.inner_view_loss(gph_emb_hid, rel_emb_hid, att_emb_hid, img_emb_hid, name_emb_hid, char_emb_hid, batch)
        loss_all = loss_joi + in_loss + out_loss

        # TODO: output_attentions
        # weight_raw = self.multimodal_encoder.fusion.weight.reshape(-1).tolist()
        loss_dic = {"joint_Intra_modal": loss_joi.item(), "Intra_modal": in_loss.item()}
        output = {"loss_dic": loss_dic, "emb": joint_emb}
        # output = {"loss_dic": loss_dic, "emb": joint_emb, "weight": weight_raw}
        return loss_all, output

    def generate_hidden_emb(self, hidden):
        i = 0
        if self.args.w_gcn:
            gph_emb = F.normalize(hidden[:, i, :].squeeze(1))
            i += 1
        else:
            gph_emb = None
        if self.args.w_rel:
            rel_emb = F.normalize(hidden[:, i, :].squeeze(1))
            i += 1
        else:
            rel_emb = None

        if self.args.w_attr:
            att_emb = F.normalize(hidden[:, i, :].squeeze(1))
            i += 1
        else:
            att_emb = None
        if self.args.w_img:
            img_emb = F.normalize(hidden[:, i, :].squeeze(1))
            i += 1
        else:
            img_emb = None

        # if hidden.shape[1] >= 6:
        if self.args.w_name and self.args.w_char:
            name_emb = F.normalize(hidden[:, i, :].squeeze(1))
            char_emb = F.normalize(hidden[:, i + 1, :].squeeze(1))
        else:
            name_emb, char_emb = None, None
            loss_name, loss_char = None, None
        emb_list = [emb for emb in [gph_emb, rel_emb, att_emb, img_emb, name_emb, char_emb] if emb is not None]
        joint_emb = torch.cat(emb_list, dim=1)

        return gph_emb, rel_emb, att_emb, img_emb, name_emb, char_emb, joint_emb

    def inner_view_loss(self, gph_emb, rel_emb, att_emb, img_emb, name_emb, char_emb, train_ill):
        # pdb.set_trace()
        loss_GCN = self.criterion_cl(gph_emb, train_ill) if gph_emb is not None else 0
        loss_rel = self.criterion_cl(rel_emb, train_ill) if rel_emb is not None else 0
        loss_att = self.criterion_cl(att_emb, train_ill) if att_emb is not None else 0
        loss_img = self.criterion_cl(img_emb, train_ill) if img_emb is not None else 0
        loss_name = self.criterion_cl(name_emb, train_ill) if name_emb is not None else 0
        loss_char = self.criterion_cl(char_emb, train_ill) if char_emb is not None else 0

        total_loss = self.multi_loss_layer([loss_GCN, loss_rel, loss_att, loss_img, loss_name, loss_char])
        return total_loss

    # def kl_alignment_loss(self, joint_emb, gph_emb, rel_emb, att_emb, img_emb, name_emb, char_emb, train_ill):
    #     zoom = self.args.zoom
    #     loss_GCN = self.criterion_align(gph_emb, joint_emb, train_ill) if gph_emb is not None else 0
    #     loss_rel = self.criterion_align(rel_emb, joint_emb, train_ill) if rel_emb is not None else 0
    #     loss_att = self.criterion_align(att_emb, joint_emb, train_ill) if att_emb is not None else 0
    #     loss_img = self.criterion_align(img_emb, joint_emb, train_ill) if img_emb is not None else 0
    #     loss_name = self.criterion_align(name_emb, joint_emb, train_ill) if name_emb is not None else 0
    #     loss_char = self.criterion_align(char_emb, joint_emb, train_ill) if char_emb is not None else 0

    #     total_loss = self.align_multi_loss_layer(
    #         [loss_GCN, loss_rel, loss_att, loss_img, loss_name, loss_char]) * zoom
    #     return total_loss

    # --------- necessary ---------------

    def joint_emb_generat(self, only_joint=True):
        # gph_emb, img_emb, rel_emb, att_emb, \
        #     name_emb, char_emb, joint_emb = self.multimodal_encoder(self.input_idx,
        if self.args.add_noise and self.multimodal_encoder.training:
            gph_emb, img_emb, rel_emb, att_emb, \
                name_emb, char_emb, joint_emb, hidden_states = self.multimodal_encoder(self.input_idx,
                                                                                    self.adj,
                                                                                    self.img_noisy_features,
                                                                                    self.rel_noisy_features,
                                                                                    self.att_noisy_features,
                                                                                    self.name_features,
                                                                                    self.char_features,
                                                                                    entity_noise=self.entity_noise, 
                                                                                    entity_noise_mask=self.entity_noise_mask)
        else:
            gph_emb, img_emb, rel_emb, att_emb, \
                name_emb, char_emb, joint_emb, hidden_states = self.multimodal_encoder(self.input_idx,
                                                                                    self.adj,
                                                                                    self.img_features,
                                                                                    self.rel_features,
                                                                                    self.att_features,
                                                                                    self.name_features,
                                                                                    self.char_features)
        if only_joint:
            return joint_emb
        else:
            return gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb, joint_emb, hidden_states
            # return gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb, joint_emb

    # --------- share ---------------

    def _get_img_dim(self, kgs):
        if isinstance(kgs["images_list"], list):
            img_dim = kgs["images_list"][0].shape[1]
        elif isinstance(kgs["images_list"], np.ndarray) or torch.is_tensor(kgs["images_list"]):
            img_dim = kgs["images_list"].shape[1]
        return img_dim

    
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
        if (epoch + 1) % (self.args.semi_learn_step * 5) == self.args.semi_learn_step:
            
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

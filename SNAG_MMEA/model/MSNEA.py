import types
import torch
import transformers
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import numpy as np
import pdb
import math
from .MSNEA_tools import AttrEncoder, generate_relation_triple_batch
from .MSNEA_loss import ContrastiveLoss
from src.utils import pairwise_distances
import os.path as osp
import json

class MSNEA(nn.Module):
    def __init__(self, kgs, args):
        super().__init__()
        self.kgs = kgs
        self.args = args
        self.ent_num = kgs["ent_num"]
        self.rel_num = kgs["rel_num"]

        self.img_embed = nn.Embedding.from_pretrained(F.normalize(torch.FloatTensor(kgs["images_list"])))
        # self.img_embed = F.normalize(torch.FloatTensor(kgs["images_list"])).cuda()
        # self.attr_embed = nn.Embedding.from_pretrained(torch.FloatTensor(kgs["att_features"]))
        self.ent_embed = nn.Embedding(self.ent_num, self.args.dim)
        self.rel_embed = nn.Embedding(self.rel_num, self.args.dim)
        nn.init.xavier_normal_(self.ent_embed.weight.data)
        nn.init.xavier_normal_(self.rel_embed.weight.data)
        self.char_dim = kgs["char_features"].shape[1] if kgs["char_features"] is not None else 100
        self.img_dim = self._get_img_dim(kgs)

        
        self.fc1 = nn.Linear(self.img_dim, self.args.dim)
        # self.fc1 = nn.Linear(2048, self.args.dim)
        # self.fc2 = nn.Linear(2048, self.args.dim)
        
        self.fc3 = nn.Linear(self.img_dim, self.args.dim)

        nn.init.xavier_normal_(self.fc1.weight.data)
        # nn.init.xavier_normal_(self.fc2.weight.data)
        nn.init.xavier_normal_(self.fc3.weight.data)

        self.attr_encoder = AttrEncoder(self.kgs, self.args)

        self.input_idx = kgs["input_idx"].cuda()

        self.name_features = None
        self.char_features = None
        if self.args.w_char and self.args.w_name:
            char_dim = kgs["char_features"].shape[1] if kgs["char_features"] is not None else 100
            self.name_fc = nn.Linear(300, self.args.char_dim)
            # pdb.set_trace()
            self.char_fc = nn.Linear(char_dim, self.args.char_dim)
            if kgs["name_features"] is not None:
                self.name_features = kgs["name_features"].cuda()
                self.char_features = kgs["char_features"].cuda()

        self.align_criterion = ContrastiveLoss()
        self.msnea_need = kgs["msnea_need"]
        self.train_e1 = torch.LongTensor(self.msnea_need["train_entities1"]).cuda()
        self.train_e2 = torch.LongTensor(self.msnea_need["train_entities2"]).cuda()

        # e1_attr = torch.LongTensor([kgs.eid_aid_list[x] for x in kgs.train_entities1]).cuda()
        # e1_val = torch.FloatTensor([kgs.eid_vid_list[x] for x in kgs.train_entities1]).cuda()
        # e2_attr = torch.LongTensor([kgs.eid_aid_list[x] for x in kgs.train_entities2]).cuda()
        # e2_val = torch.FloatTensor([kgs.eid_vid_list[x] for x in kgs.train_entities2]).cuda()
        # mask1 = torch.ByteTensor([kgs.eid_mask_list[x] for x in kgs.train_entities1]).cuda()
        # mask2 = torch.ByteTensor([kgs.eid_mask_list[x] for x in kgs.train_entities2]).cuda()
        # l1 = torch.FloatTensor([kgs.eav_len_list[x] for x in kgs.train_entities1]).cuda()
        # l2 = torch.FloatTensor([kgs.eav_len_list[x] for x in kgs.train_entities2]).cuda()
        # self.label_ = torch.eye(len(self.train_e1)).cuda()
        self.step = 0

    def forward(self, batch):
        
        bs = self.args.batch_size
        bs = batch.shape[0]
        batch_pos, batch_neg = generate_relation_triple_batch(self.msnea_need["relation_triples_list1"],
                                                              self.msnea_need["relation_triples_list2"], self.msnea_need["relation_triples_set1"],
                                                              self.msnea_need["relation_triples_set2"], self.msnea_need["kg1_entities_list"],
                                                              self.msnea_need["kg2_entities_list"], bs, self.step, self.args.neg_triple_num)
        self.step += 1
        # batch_pos, batch_neg = generate_relation_triple_batch(batch, self.msnea_need["relation_triples_set"], self.msnea_need["entities_list"], bs, self.args.neg_triple_num)

        rel_p_h = torch.LongTensor([x[0] for x in batch_pos]).cuda()
        rel_p_r = torch.LongTensor([x[1] for x in batch_pos]).cuda()
        rel_p_t = torch.LongTensor([x[2] for x in batch_pos]).cuda()
        rel_n_h = torch.LongTensor([x[0] for x in batch_neg]).cuda()
        rel_n_r = torch.LongTensor([x[1] for x in batch_neg]).cuda()
        rel_n_t = torch.LongTensor([x[2] for x in batch_neg]).cuda()
        
        # pdb.set_trace()
        train_e1 = torch.LongTensor(batch[:, 0]).cuda()
        train_e2 = torch.LongTensor(batch[:, 1]).cuda()
        label_ = torch.eye(bs).cuda()
        # pdb.set_trace()
        r_loss, rs, ats, ims, score = self.kge(rel_p_h, rel_p_r, rel_p_t, rel_n_h, rel_n_r, rel_n_t, train_e1, train_e2)
        
        align_loss = self.align_criterion(score, label_) + self.align_criterion(rs, label_) + self.align_criterion(ats, label_) + self.align_criterion(ims, label_)
        loss = r_loss + align_loss

        loss_dic = {"kge": r_loss.item(), "align": align_loss.item()}
        output = {"loss_dic": loss_dic}
        return loss, output

    def kge(self, p_h, p_r, p_t, n_h, n_r, n_t, e1, e2):

        r_p_h = self.r_rep(p_h)
        i_p_h = self.i_w(p_h)
        r_p_r = F.normalize(self.rel_embed(p_r), 2, -1)
        r_p_t = self.r_rep(p_t)
        i_p_t = self.i_w(p_t)

        r_n_h = self.r_rep(n_h)
        i_n_h = self.i_w(n_h)
        r_n_r = F.normalize(self.rel_embed(n_r), 2, -1)
        r_n_t = self.r_rep(n_t)
        i_n_t = self.i_w(n_t)

        pos_dis = r_p_h + r_p_r - r_p_t
        neg_dis = r_n_h + r_n_r - r_n_t

        pos_score = torch.sum(torch.square(pos_dis), dim=1)
        neg_score = torch.sum(torch.square(neg_dis), dim=1)
        relation_loss = torch.sum(F.relu(self.args.margin + pos_score - neg_score))
        pos_dis1 = i_p_h + r_p_r - i_p_t
        neg_dis1 = i_n_h + r_n_r - i_n_t
        pos_score1 = torch.sum(torch.square(pos_dis1), dim=1)
        neg_score1 = torch.sum(torch.square(neg_dis1), dim=1)
        relation_loss += torch.sum(F.relu(self.args.margin + pos_score1 - neg_score1))

        
        e1_i_embed, e1_r_embed, e1_a_embed, name_emb, char_emb = self._emb_generate(e1, self.name_features, self.char_features)
        e2_i_embed, e2_r_embed, e2_a_embed, name_emb, char_emb = self._emb_generate(e2, self.name_features, self.char_features)

        # e1_a_embed = self.attr_encoder(e1, e1_i_embed)
        # e2_a_embed = self.attr_encoder(e2, e2_i_embed)

        e1_all = self.fusion([e1_r_embed, e1_i_embed, e1_a_embed, name_emb, char_emb])
        e2_all = self.fusion([e2_r_embed, e2_i_embed, e2_a_embed, name_emb, char_emb])

        r_score = torch.mm(e1_r_embed, e2_r_embed.t())
        a_score = torch.mm(e1_a_embed, e2_a_embed.t())
        i_score = torch.mm(e1_i_embed, e2_i_embed.t())
        score = torch.mm(e1_all, e2_all.t())

        return relation_loss, r_score, a_score, i_score, score

    def r_rep(self, e):
        return F.normalize(self.ent_embed(e), 2, -1)

    def i_rep(self, e):
        return F.normalize(self.fc1(self.img_embed(e)), 2, -1)

    def i_w(self, e):

        return F.normalize(self.fc3(self.img_embed(e)), 2, -1)

    def fusion(self, embs):

        embs = [F.normalize(embs[idx]) for idx in range(len(embs)) if embs[idx] is not None]
        all = F.normalize(torch.cat(embs, dim=1), 2, -1)
        return all

    # --------- necessary ---------------
    def joint_emb_generat(self):
        # ablation
        img_emb, rel_emb, att_emb, name_emb, char_emb = self._emb_generate(self.input_idx,
                                                                           self.name_features,
                                                                           self.char_features)

        joint_emb = self.fusion([rel_emb, img_emb, att_emb, name_emb, char_emb])

        # if self.args.w_name and self.args.w_char:
        #     joint_emb = torch.cat([
        #         F.normalize(rel_emb).detach(),
        #         F.normalize(img_emb).detach(),
        #         F.normalize(att_emb).detach(),
        #         F.normalize(att_emb).detach(),
        #         F.normalize(char_emb).detach(),
        #     ], dim=1)
        # else:
        #     joint_emb = torch.cat([
        #         F.normalize(rel_emb).detach(),
        #         F.normalize(img_emb).detach(),
        #         F.normalize(att_emb).detach(),
        #     ], dim=1)

        return joint_emb

    def _emb_generate(self, input_idx, name_features=None, char_features=None):
        if self.args.w_img:
            img_emb = self.i_rep(input_idx)
        else:
            img_emb = None
        if self.args.w_rel:
            rel_emb = self.r_rep(input_idx)
        else:
            rel_emb = None
        if self.args.w_attr and self.args.w_img:
            att_emb = self.attr_encoder(input_idx, img_emb)
        else:
            att_emb = None
        if self.args.w_name:
            name_emb = self.name_fc(name_features[input_idx])
        else:
            name_emb = None
        if self.args.w_char:
            char_emb = self.char_fc(char_features[input_idx])
        else:
            char_emb = None

        return (img_emb, rel_emb, att_emb, name_emb, char_emb)

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

            if self.args.rank == 0:
                logger.info(f"#new_links_select:{len(new_links_select)}")
                logger.info(f"train_ill.shape:{train_ill.shape}")
                logger.info(f"#true_links: {num_true}")
                logger.info(f"true link ratio: {(100 * num_true / len(new_links_select)):.1f}%")
                logger.info(f"#entity not in train set: {len(left_non_train)} (left) {len(right_non_train)} (right)")

            new_links = []
        else:
            logger.info("len(new_links) is 0")

        return left_non_train, right_non_train, train_ill, new_links

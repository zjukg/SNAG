import os
import os.path as osp
import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler
from torch.cuda.amp import GradScaler, autocast
from datetime import datetime
from easydict import EasyDict as edict
from tqdm import tqdm
import pdb
import pprint
import json
import pickle
from collections import defaultdict

from config import cfg
from torchlight import initialize_exp, set_seed, get_dump_path
from src.data import load_data, Collator_base, EADataset
from src.data_msnea import load_msnea_data
from src.utils import set_optim, Loss_log, pairwise_distances, csls_sim
from model import EVA, MCLEA, MEAformer, MSNEA, SNAG

import torch.nn.functional as F
# EVA
import scipy
import gc
import copy

class Runner:
    def __init__(self, args, writer=None, logger=None):
        self.datapath = edict()
        self.datapath.log_dir = get_dump_path(args)
        self.datapath.model_dir = os.path.join(self.datapath.log_dir, 'model')
        self.args = args
        self.writer = writer
        self.logger = logger
        self.scaler = GradScaler()
        self.model_list = []
        set_seed(args.random_seed)
        self.data_init()
        self.model_choise()
        set_seed(args.random_seed)

        if self.args.only_test:
            self.dataloader_init(test_set=self.test_set)
        else:
            self.dataloader_init(train_set=self.train_set, eval_set=self.eval_set, test_set=self.test_set)
            self.model_list = [self.model]
            if self.args.il:
                assert self.args.il_start < self.args.epoch
                train_epoch_1_stage = self.args.il_start
            else:
                train_epoch_1_stage = self.args.epoch
            self.optim_init(self.args, total_epoch=train_epoch_1_stage)

    def model_choise(self):
        assert self.args.model_name in ["EVA", "MCLEA", "MSNEA", "MEAformer", "SNAG"]
        if self.args.model_name == "EVA":
            self.model = EVA(self.KGs, self.args)
        elif self.args.model_name == "MCLEA":
            self.model = MCLEA(self.KGs, self.args)
        elif self.args.model_name == "MSNEA":
            self.model = MSNEA(self.KGs, self.args)
        elif self.args.model_name == "MEAformer":
            self.model = MEAformer(self.KGs, self.args)
        elif self.args.model_name == "SNAG":
            self.model = SNAG(self.KGs, self.args)

        self.model = self._load_model(self.model)

        # To be check
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"total params num: {total_params}")

    def optim_init(self, opt, total_step=None, total_epoch=None, accumulation_step=None):
        step_per_epoch = len(self.train_dataloader)
        if total_epoch is not None:
            opt.total_steps = int(step_per_epoch * total_epoch)
        else:
            opt.total_steps = int(step_per_epoch * opt.epoch) if total_step is None else int(total_step)
        opt.warmup_steps = int(opt.total_steps * 0.15)

        if total_step is None:
            self.logger.info(f"warmup_steps: {opt.warmup_steps}")
            self.logger.info(f"total_steps: {opt.total_steps}")
            self.logger.info(f"weight_decay: {opt.weight_decay}")
        freeze_part = []
        no_freeze_part = []

        self.optimizer, self.scheduler = set_optim(opt, self.model_list, freeze_part, no_freeze_part, accumulation_step)

    def data_init(self):
        if self.args.model_name != "MSNEA":
            self.KGs, self.non_train, self.train_set, self.eval_set, self.test_set, self.test_ill_ = load_data(self.logger, self.args)
        else:
            self.KGs, self.non_train, self.train_set, self.eval_set, self.test_set, self.test_ill_ = load_msnea_data(self.logger, self.args)
        self.train_ill = self.train_set.data
        self.eval_left = torch.LongTensor(self.eval_set[:, 0].squeeze()).cuda()
        self.eval_right = torch.LongTensor(self.eval_set[:, 1].squeeze()).cuda()
        if self.test_set is not None:
            self.test_left = torch.LongTensor(self.test_ill[:, 0].squeeze()).cuda()
            self.test_right = torch.LongTensor(self.test_ill[:, 1].squeeze()).cuda()

        self.eval_sampler = None

    def dataloader_init(self, train_set=None, eval_set=None, test_set=None):
        bs = self.args.batch_size
        collator = Collator_base(self.args)
        self.args.workers = min([os.cpu_count(), self.args.batch_size, self.args.workers])
        if train_set is not None:
            self.train_dataloader = self._dataloader(train_set, bs, collator)
        if test_set is not None:
            self.test_dataloader = self._dataloader(test_set, bs, collator)
        if eval_set is not None:
            self.eval_dataloader = self._dataloader(eval_set, bs, collator)

    def _dataloader(self, train_set, batch_size, collator):
        train_dataloader = DataLoader(
            train_set,
            num_workers=self.args.workers,
            persistent_workers=True,  # True
            shuffle=(self.args.only_test == 0),
            # drop_last=(self.args.only_test == 0),
            drop_last=False,
            batch_size=batch_size,
            collate_fn=collator
        )
        # pdb.set_trace()
        return train_dataloader

    def run(self):
        self.loss_log = Loss_log()
        self.curr_loss = 0.
        self.lr = self.args.lr
        self.curr_loss_dic = defaultdict(float)
        self.weight = [1, 1, 1, 1, 1, 1]
        self.loss_weight = [1, 1]
        self.loss_item = 99999.
        self.step = 1
        self.epoch = 0
        self.new_links = []
        self.best_model_wts = None

        self.best_mrr = 0

        self.early_stop_init = 200
        self.early_stop_count = self.early_stop_init
        self.stage = 0  

        with tqdm(total=self.args.epoch) as _tqdm:  
            for i in range(self.args.epoch):
                # _tqdm.set_description(f'Train | epoch {i} Loss {self.loss_log.get_loss():.5f} Acc {self.loss_log.get_acc()*100:.3f}%')
                # -------------------------------
                self.epoch = i
                
                if self.args.il and ((self.epoch == self.args.il_start and self.stage == 0) or (self.early_stop_count <= 0 and self.epoch <= self.args.il_start)):
                    if self.early_stop_count <= 0:
                        logger.info(f"Early stop in epoch {self.epoch}... Begin iteration....")
                    self.stage = 1
                    self.early_stop_init = 200
                    self.early_stop_count = self.early_stop_init

                    self.step = 1
                    self.args.lr = self.args.lr / 5
                    
                    self.optim_init(self.args, total_epoch=(self.args.epoch - self.args.il_start) * 3)
                    
                    if self.best_model_wts is not None:
                        self.logger.info("load from the best model before IL... ")
                        self.model.load_state_dict(self.best_model_wts)
                    
                    name = self._save_name_define()
                    self.test(save_name=f"{name}_test_ep{self.args.epoch}_no_iter")

                
                if self.stage == 1 and (self.epoch + 1) % self.args.semi_learn_step == 0 and self.args.il:
                    self.il_for_ea()

                if self.stage == 1 and (self.epoch + 1) % (self.args.semi_learn_step * 10) == 0 and len(self.new_links) != 0 and self.args.il:
                    self.il_for_data_ref()
                    # pass

                self.train(_tqdm)
                
                self.loss_log.update(self.curr_loss)
                self.loss_item = self.loss_log.get_loss()
                _tqdm.set_description(f'Train | Ep [{self.epoch}/{self.args.epoch}] Step [{self.step}/{self.args.total_steps}] LR [{self.lr:.5f}] Loss {self.loss_log.get_loss():.5f} ')
                self.update_loss_log()
                # writer.add_scalars("name",{"dic":val}, epoch)
                if (i + 1) % self.args.eval_epoch == 0:
                    self.eval()
                # https://zhuanlan.zhihu.com/p/382950853
                # writer.add_scalars("loss",{"dic":val}, epoch)
                _tqdm.update(1)
                if self.stage == 1 and self.early_stop_count <= 0:
                    logger.info(f"Early stop in epoch {self.epoch}")
                    break

        name = self._save_name_define()
        # self.eval(last_epoch=True, save_name=f"{name}_eval_ep{self.args.epoch}")
        if self.best_model_wts is not None:
            self.logger.info("load from the best model before final testing ... ")
            self.model.load_state_dict(self.best_model_wts)
        self.test(save_name=f"{name}_test_ep{self.args.epoch}")

        # TODO: save or load
        self.logger.info(f"min loss {self.loss_log.get_min_loss()}")
        if not self.args.only_test and self.args.save_model:
            self._save_model(self.model, input_name=name)

    
    def il_for_ea(self):
        with torch.no_grad():
            if self.args.model_name in ["SNAG"]:
                final_emb, weight_norm = self.model.joint_emb_generat()
            else:
                final_emb = self.model.joint_emb_generat()
            final_emb = F.normalize(final_emb)
            self.new_links = self.model.Iter_new_links(self.epoch, self.non_train["left"], final_emb, self.non_train["right"], new_links=self.new_links)
            if (self.epoch + 1) % (self.args.semi_learn_step * 5) == 0:
                self.logger.info(f"[epoch {self.epoch}] #links in candidate set: {len(self.new_links)}")

    
    def il_for_data_ref(self):
        
        # pdb.set_trace()
        self.non_train["left"], self.non_train["right"], self.train_ill, self.new_links = self.model.data_refresh(
            self.logger, self.train_ill, self.test_ill_, self.non_train["left"], self.non_train["right"], new_links=self.new_links)
        
        # pdb.set_trace()
        set_seed(self.args.random_seed)
        self.train_set = EADataset(self.train_ill)
        self.dataloader_init(train_set=self.train_set)

        # one time train

    def _save_name_define(self):
        prefix = ""
        if self.args.il:
            prefix = f"il{self.args.epoch-self.args.il_start}_b{self.args.il_start}_{prefix}"
        name = f'{self.args.exp_id}_{prefix}'
        return name

    def train(self, _tqdm):
        self.model.train()
        curr_loss = 0.
        self.loss_log.acc_init()
        accumulation_steps = self.args.accumulation_steps
        # torch.cuda.empty_cache()
        
        if self.args.add_noise:
            self.model.update_noise()

        for batch in self.train_dataloader:
            # with autocast():

            loss, output = self.model(batch)
            loss = loss / accumulation_steps
            self.scaler.scale(loss).backward()
            self.step += 1
            # pdb.set_trace()

            curr_loss += loss.item()
            self.output_statistic(loss, output)

            if self.step % accumulation_steps == 0:
                
                self.scaler.unscale_(self.optimizer)
                for model in self.model_list:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip)
                scale = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                skip_lr_sched = (scale > self.scaler.get_scale())
                if not skip_lr_sched:
                    # pdb.set_trace()
                    self.scheduler.step()

                # pdb.set_trace()
                self.lr = self.scheduler.get_last_lr()[-1]
                self.writer.add_scalars("lr", {"lr": self.lr}, self.step)
                
                # pdb.set_trace()
                for model in self.model_list:
                    model.zero_grad(set_to_none=True)

        return curr_loss

    def output_statistic(self, loss, output):
        
        self.curr_loss += loss.item()
        if output is None:
            return
        for key in output['loss_dic'].keys():
            self.curr_loss_dic[key] += output['loss_dic'][key]
        # pdb.set_trace()
        if 'weight' in output and output['weight'] is not None:
            self.weight = output['weight']
        if 'loss_weight' in output and output['loss_weight'] is not None:
            self.loss_weight = output['loss_weight']

    def update_loss_log(self):
        
        # https://zhuanlan.zhihu.com/p/382950853
        #  "mask_loss": self.curr_loss_dic['mask_loss'], "ke_loss": self.curr_loss_dic['ke_loss']
        vis_dict = {"train_loss": self.curr_loss}
        vis_dict.update(self.curr_loss_dic)
        self.writer.add_scalars("loss", vis_dict, self.step)
        

        if self.weight is not None:
            weight_dic = {}
            weight_dic["img"] = self.weight[0]
            weight_dic["attr"] = self.weight[1]
            weight_dic["rel"] = self.weight[2]
            weight_dic["graph"] = self.weight[3]
            if self.args.w_name or self.args.w_char:
                weight_dic["name"] = self.weight[4]
                weight_dic["char"] = self.weight[5]
            self.writer.add_scalars("modal_weight", weight_dic, self.step)

        if self.loss_weight is not None and self.loss_weight != [1, 1]:
            weight_dic = {}
            weight_dic["mask"] = 1 / (self.loss_weight[0]**2)
            weight_dic["kpi"] = 1 / (self.loss_weight[1]**2)
            self.writer.add_scalars("loss_weight", weight_dic, self.step)
            # vis_kpi_dic = {"recover": 1 / (self.kpi_weight[0]**2), "classifier": 1 / (self.kpi_weight[1]**2)}
            # if self.args.contrastive_loss and len(self.kpi_weight) > 2:
            #     vis_kpi_dic.update({"contrastive": 1 / (self.kpi_weight[2]**2)})
            # self.writer.add_scalars("kpi_weight", vis_kpi_dic, self.step)

        # init log loss
        self.curr_loss = 0.
        for key in self.curr_loss_dic:
            self.curr_loss_dic[key] = 0.

    # one time eval
    def eval(self, last_epoch=False, save_name=""):
        test_left = self.eval_left
        test_right = self.eval_right
        self.model.eval()
        self._test(test_left, test_right, last_epoch=last_epoch, save_name=save_name)
        # torch.cuda.empty_cache()

    # one time test
    def test(self, save_name=""):
        if self.test_set is None:
            test_left = self.eval_left
            test_right = self.eval_right
        else:
            test_left = self.test_left
            test_right = self.test_right
        self.model.eval()
        self.logger.info(" --------------------- Test result --------------------- ")
        self._test(test_left, test_right, last_epoch=True, save_name=save_name)

    def _test(self, test_left, test_right, last_epoch=False, save_name="", loss=None):
        with torch.no_grad():
            if self.args.model_name in ["EVA", "MCLEA"]:
                if self.args.model_name == "EVA":
                    self.model.emb_generat()
                    w_normalized = F.softmax(self.model.weight_raw, dim=0)
                else:
                    
                    # MCLEA
                    w_normalized = F.softmax(self.model.multimodal_encoder.fusion.weight.reshape(-1), dim=0)
                # pdb.set_trace()
                appdx = ""
                if self.args.w_name and self.args.w_char:
                    appdx = f"-[name_{w_normalized[4]:.3f}]-[char_{w_normalized[5]:.3f}]"
                self.logger.info(f"weight_raw:[img_{w_normalized[0]:.3f}]-[attr_{w_normalized[1]:.3f}]-[rel_{w_normalized[2]:.3f}]-[graph_{w_normalized[3]:.3f}]{appdx}")

            if self.args.model_name in ["SNAG"]:
                final_emb, weight_norm = self.model.joint_emb_generat()
            else:
                final_emb = self.model.joint_emb_generat()
            final_emb = F.normalize(final_emb)
        top_k = [1, 10, 50]

        acc_l2r = np.zeros((len(top_k)), dtype=np.float32)
        acc_r2l = np.zeros((len(top_k)), dtype=np.float32)
        test_total, test_loss, mean_l2r, mean_r2l, mrr_l2r, mrr_r2l = 0, 0., 0., 0., 0., 0.
        if self.args.distance == 2:
            distance = pairwise_distances(final_emb[test_left], final_emb[test_right])
        elif self.args.distance == 1:
            distance = torch.FloatTensor(scipy.spatial.distance.cdist(
                final_emb[test_left].cpu().data.numpy(),
                final_emb[test_right].cpu().data.numpy(), metric="cityblock"))

        if self.args.csls is True:
            distance = 1 - csls_sim(1 - distance, self.args.csls_k)

        if last_epoch:
            to_write = []
            test_left_np = test_left.cpu().numpy()
            test_right_np = test_right.cpu().numpy()
            to_write.append(["idx", "rank", "query_id", "gt_id", "ret1", "ret2", "ret3"])
        for idx in range(test_left.shape[0]):
            values, indices = torch.sort(distance[idx, :], descending=False)
            rank = (indices == idx).nonzero(as_tuple=False).squeeze().item()
            mean_l2r += (rank + 1)
            mrr_l2r += 1.0 / (rank + 1)
            for i in range(len(top_k)):
                if rank < top_k[i]:
                    acc_l2r[i] += 1
            # save idx, correct rank pos, and indices
            if last_epoch:
                indices = indices.cpu().numpy()
                to_write.append([idx, rank, test_left_np[idx], test_right_np[idx], test_right_np[indices[0]], test_right_np[indices[1]], test_right_np[indices[2]]])
        if last_epoch:
            import csv
            if save_name == "":
                save_name = self.args.model_name
            save_pred_path = osp.join(self.args.data_path, self.args.model_name, f"{save_name}_pred")
            os.makedirs(save_pred_path, exist_ok=True)
            with open(osp.join(save_pred_path, f"{self.args.data_choice}_pred.txt"), "w") as f:
                wr = csv.writer(f, dialect='excel')
                wr.writerows(to_write)

        for idx in range(test_right.shape[0]):
            _, indices = torch.sort(distance[:, idx], descending=False)
            rank = (indices == idx).nonzero(as_tuple=False).squeeze().item()
            mean_r2l += (rank + 1)
            mrr_r2l += 1.0 / (rank + 1)
            for i in range(len(top_k)):
                if rank < top_k[i]:
                    acc_r2l[i] += 1
        mean_l2r /= test_left.size(0)
        mean_r2l /= test_right.size(0)
        mrr_l2r /= test_left.size(0)
        mrr_r2l /= test_right.size(0)
        for i in range(len(top_k)):
            acc_l2r[i] = round(acc_l2r[i] / test_left.size(0), 4)
            acc_r2l[i] = round(acc_r2l[i] / test_right.size(0), 4)
        gc.collect()

        Loss_out = f", Loss = {self.loss_item:.4f}"
        self.logger.info(f"Ep {self.epoch} | l2r: acc of top {top_k} = {acc_l2r}, mr = {mean_l2r:.3f}, mrr = {mrr_l2r:.3f}{Loss_out}")
        self.logger.info(f"Ep {self.epoch} | r2l: acc of top {top_k} = {acc_r2l}, mr = {mean_r2l:.3f}, mrr = {mrr_r2l:.3f}{Loss_out}")
        if last_epoch:
            t1, t2, t3 = acc_l2r
            self.logger.info(f"Res:[{t1}\t{t2}\t{mrr_l2r:.3f}]")

        # pdb.set_trace()
        self.early_stop_count -= 1
        if mrr_l2r > max(self.loss_log.acc) and not last_epoch:
            self.logger.info(f"Best model update in Ep {self.epoch}: MRR from [{max(self.loss_log.acc)}] --> [{mrr_l2r}] ... ")
            self.loss_log.update_acc(mrr_l2r)
            self.early_stop_count = self.early_stop_init

            # deep copy
            # best_model_wts = copy.deepcopy(model.state_dict())
            self.best_model_wts = copy.deepcopy(self.model.state_dict())

    def _load_model(self, model, model_name=None):
        # TODO: path
        if model_name is None:
            model_name = self.args.model_name_save
        save_path = osp.join(self.args.data_path, self.args.model_name, 'save')
        save_path = osp.join(save_path, f'{model_name}.pkl')
        if (len(model_name) == 0 or not os.path.exists(save_path)):
            if len(model_name) > 0:
                self.logger.info(f"{model_name}.pkl not exist!!")
            else:
                self.logger.info("Random init...")
            model.cuda()
            return model

        if 'Dist' in self.args.model_name:
            model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(save_path, map_location=self.args.device).items()})
        else:
            model.load_state_dict(torch.load(save_path, map_location=self.args.device))

        model.cuda()
        self.logger.info(f"loading model [{model_name}.pkl] done!")

        return model

    def _save_model(self, model, input_name=""):

        model_name = self.args.model_name
        # TODO: path

        save_path = osp.join(self.args.data_path, model_name, 'save')
        os.makedirs(save_path, exist_ok=True)

        if input_name == "":
            input_name = self._save_name_define()
        save_path = osp.join(save_path, f'{input_name}.pkl')

        if model is None:
            return
        if self.args.save_model:
            torch.save(model.state_dict(), save_path)

            self.logger.info(f"saving [{save_path}] done!")

        return save_path

if __name__ == '__main__':
    cfg = cfg()
    cfg.get_args()
    cfgs = cfg.update_train_configs()
    set_seed(cfgs.random_seed)
    # -----  Init ----------
    torch.multiprocessing.set_sharing_strategy('file_system')
    writer, logger = None, None
    logger = initialize_exp(cfgs)
    logger_path = get_dump_path(cfgs)
    cfgs.time_stamp = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    comment = f'bath_size={cfgs.batch_size} exp_id={cfgs.exp_id}'
    if not cfgs.no_tensorboard and not cfgs.only_test:
        writer = SummaryWriter(log_dir=os.path.join(logger_path, 'tensorboard', cfgs.time_stamp), comment=comment)

    cfgs.device = torch.device(cfgs.gpu)
    # -----  Begin ----------
    torch.cuda.set_device(cfgs.gpu)
    runner = Runner(cfgs, writer, logger)
    if cfgs.only_test:
        runner.test(last_epoch=False)
    else:
        runner.run()

    # -----  End ----------
    if not cfgs.no_tensorboard and not cfgs.only_test:
        writer.close()
        logger.info("done!")

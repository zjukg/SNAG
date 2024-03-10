
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import math

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, dis, label, margin=2.0):
        loss_contrastive = torch.mean((1 - label) * torch.pow(dis, 2) +
                                      (label) * torch.pow(torch.clamp(margin - dis, min=0.0), 2))
        return loss_contrastive

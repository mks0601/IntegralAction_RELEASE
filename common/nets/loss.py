import torch
import torch.nn as nn
from torch.nn import functional as F

class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, label_out, label_target):
        loss = self.ce_loss(label_out, label_target)
        return loss

class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bce_loss = nn.BCELoss(reduction='none')

    def forward(self, label_out, label_target):
        loss = self.bce_loss(label_out, label_target)
        return loss



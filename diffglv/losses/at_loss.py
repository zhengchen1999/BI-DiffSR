import torch
from torch import nn as nn
from torch.nn import functional as F
from basicsr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class ATLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(ATLoss, self).__init__()


    def forward(self, pred, target):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
        """
        Attention_pred = F.normalize(pred.pow(2).mean(1).view(pred.size(0), -1)) # (N, H*W)
        Attention_target = F.normalize(target.pow(2).mean(1).view(target.size(0), -1)) # (N, H*W)

        return  nn.MSELoss()(Attention_pred, Attention_target)

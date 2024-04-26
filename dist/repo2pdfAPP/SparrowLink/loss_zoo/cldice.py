import torch
import torch.nn as nn
import torch.nn.functional as F
from .soft_skeleton import soft_skel
from monai.losses import DiceLoss
from monai.networks import one_hot


class soft_cldice(nn.Module):
    def __init__(self, iter_=3, smooth = 1.):
        super(soft_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth

    def forward(self, y_true, y_pred):
        skel_pred = soft_skel(y_pred, self.iter)
        skel_true = soft_skel(y_true, self.iter)
        tprec = (torch.sum(torch.multiply(skel_pred, y_true)[:,1:,...])+self.smooth)/(torch.sum(skel_pred[:,1:,...])+smooth)
        tsens = (torch.sum(torch.multiply(skel_true, y_pred)[:,1:,...])+self.smooth)/(torch.sum(skel_true[:,1:,...])+smooth)
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
        return cl_dice


def soft_dice(y_true, y_pred):
    """[function to compute dice loss]
    Args:
        y_true ([float32]): [ground truth image]
        y_pred ([float32]): [predicted image]
    Returns:
        [float32]: [loss value]
    """
    smooth = 1
    intersection = torch.sum((y_true * y_pred)[:,1:,...])
    coeff = (2. * intersection + smooth) / (torch.sum(y_true[:,1:,...]) + torch.sum(y_pred[:,1:,...]) + smooth)
    return (1. - coeff)


class soft_dice_cldice(nn.Module):
    def __init__(self, iter_=3, alpha=0.5, smooth=1.):
        super(soft_dice_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self.alpha = alpha
        self.dice = DiceLoss(to_onehot_y=False, softmax=False)

    def forward(self, y_pred, y_true):
        y_true = one_hot(y_true, num_classes=2)
        y_pred = F.softmax(y_pred, dim=1)
        dice = self.dice(y_pred, y_true)
        skel_pred = soft_skel(y_pred, self.iter)
        skel_true = soft_skel(y_true, self.iter)
        tprec = (torch.sum(torch.multiply(skel_pred, y_true)[:,1:,...])+self.smooth)/(torch.sum(skel_pred[:,1:,...])+self.smooth)
        tsens = (torch.sum(torch.multiply(skel_true, y_pred)[:,1:,...])+self.smooth)/(torch.sum(skel_true[:,1:,...])+self.smooth)
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
        return (1.0-self.alpha)*dice+self.alpha*cl_dice


class soft_dice_cldice_weighted(nn.Module):
    def __init__(self, iter_=3, alpha=0.5, smooth = 1.):
        super(soft_dice_cldice_weighted, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self.alpha = alpha
        self.dice = DiceLoss(to_onehot_y=False, softmax=False, batch=True, reduction="none")

    def forward(self, y_pred, y_true, weight):
        # y_true = one_hot(y_true, num_classes=2)
        # y_pred = F.softmax(y_pred, dim=1)
        intersection = torch.sum((y_true * y_pred)[:, 1:, ...], dim=(1, 2, 3, 4), keepdim=True)
        coeff = (2. * intersection + self.smooth) / (torch.sum(y_true[:, 1:, ...], dim=(1, 2, 3, 4), keepdim=True) +
                                                     torch.sum(y_pred[:, 1:, ...], dim=(1, 2, 3, 4), keepdim=True) + self.smooth)
        dice = 1.0 - coeff
        skel_pred = soft_skel(y_pred, self.iter)
        skel_true = soft_skel(y_true, self.iter)
        tprec = (torch.sum(torch.multiply(skel_pred, y_true)[:,1:,...], dim=(1, 2, 3, 4), keepdim=True)+self.smooth)/(torch.sum(skel_pred[:,1:,...], dim=(1, 2, 3, 4), keepdim=True)+self.smooth)
        tsens = (torch.sum(torch.multiply(skel_true, y_pred)[:,1:,...], dim=(1, 2, 3, 4), keepdim=True)+self.smooth)/(torch.sum(skel_true[:,1:,...], dim=(1, 2, 3, 4), keepdim=True)+self.smooth)
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
        dice = torch.mean(dice * weight)
        cl_dice = torch.mean(cl_dice * weight)
        return (1.0-self.alpha)*dice+self.alpha*cl_dice
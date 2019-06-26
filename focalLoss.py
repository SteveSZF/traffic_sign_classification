import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, num_classes, cuda=1):
        super(FocalLoss, self).__init__()
        self.name = 'FocalLoss'
        self.num_cls = num_classes
        self.one_hot = torch.eye(num_classes).cuda()

    def focal_loss(self, x, y):
        alpha = 0.25
        gamma = 2
        t = self.one_hot[y.data, :]
        t = Variable(t)
        p = x.sigmoid()
        pt = p*t + (1-p)*(1-t)         # pt = p if t > 0 else 1-p
        w = alpha*t + (1-alpha)*(1-t)  # w = alpha if t > 0 else 1-alpha
        w = w * (1-pt).pow(gamma)
        w = w.detach()
        return F.binary_cross_entropy_with_logits(x, t, w, size_average=False)
    
    def forward(self, cls_pred, cls_truth):
        #calculate the classify loss
        B, _ = cls_pred.size()
        cls_truth = cls_truth.view(-1)
        loss = self.focal_loss(cls_pred, cls_truth)
        self.loss = loss / B
        return self.loss



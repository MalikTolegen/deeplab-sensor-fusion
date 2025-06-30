import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from torchvision.ops.focal_loss import sigmoid_focal_loss


class DiceFocalLoss(nn.Module):
    def __init__(self):
        super(DiceFocalLoss, self).__init__()
    
    def forward(self, inputs: Tensor, targets: Tensor, smooth: float=1.):
        inputs = F.sigmoid(inputs)
        # inputs = torch.argmax(inputs, dim=1)
        inputs = inputs.view(-1) # .to(torch.float32)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        cls_loss = F.cross_entropy(inputs, targets, reduction='mean')
        # cls_loss = sigmoid_focal_loss(inputs, targets, reduction='mean')

        losses = dict(
            cls_loss=cls_loss,
            # dice_loss=dice_loss
        )
        # loss = ce_loss + dice_loss

        return losses
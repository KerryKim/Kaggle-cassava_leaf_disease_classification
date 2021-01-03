import torch
import torch.nn.functional as F

from torch.nn.modules.loss import _WeightedLoss

##
# reference : https://dacon.io/competitions/official/235626/codeshare/1677?page=2&dtype=recent&ptype=pub
def CrossEntropyLoss(output, label):
    # ref. fn_loss = nn.CrossEntropyLoss().to(device)
    if len(label.shape) == 1:
        return F.cross_entropy(output, label)
    # if multi-labels
    else:
        return torch.mean(torch.sum(-label * F.log_softmax(output, dim=1), dim=1))

def Label_Smooth_CrossEntropyLoss(output, label, epsilon=0.1):
    n_class = output.shape[1]
    device = output.device
    onehot = F.one_hot(label, n_class).to(dtype=torch.float, device=device)
    label = (1 - epsilon) * onehot + torch.ones(onehot.shape).to(device) * epsilon / n_class
    return CrossEntropyLoss(output, label)

'''
# reference: https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/173733
class CustomCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean'):
        super().__init__(weight=weight, reduction=reduction)
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)
        loss = -(targets * lsm).sum(-1)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss

class CustomSmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets:torch.Tensor, n_classes:int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes), device=targets.device).fill_(smoothing /(n_classes-1)) \
                .scatter_(1, targets.data.unsqueeze(1), 1.-smoothing)
        return targets

    def forward(self, inputs, targets):
        targets = CustomSmoothCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1),
            self.smoothing)
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss
'''
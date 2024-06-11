
import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import json


class BalancedSoftmax(_Loss):
    """
    Balanced Softmax Loss
    """
    def __init__(self, freq_path):
        super(BalancedSoftmax, self).__init__()
        with open(freq_path, 'r') as fd:
            freq = json.load(fd)
        freq = torch.tensor(freq)
        self.sample_per_class = freq

    def forward(self, input, label, reduction='mean'):
        return balanced_softmax_loss(label, input, self.sample_per_class, reduction)


def balanced_softmax_loss(labels, logits, sample_per_class):

    loss = 0
    for idx, prediction in enumerate(logits):
        denominator = 0
        for cls in range(sample_per_class.size(0)):
            if sample_per_class[cls] != 0:
                denominator += torch.exp(prediction[cls])
        loss -= torch.log(torch.exp(prediction[labels[idx]]))
    loss = loss / labels.size(0)
    return loss


def create_loss(freq_path):
    print('Loading Balanced Softmax Loss.')
    return BalancedSoftmax(freq_path)
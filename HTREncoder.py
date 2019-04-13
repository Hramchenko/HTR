import torch
import torch.nn as nn
import torch.nn.functional as F
from Layers import *

class HTREncoder(nn.Module):
    def __init__(self, batchnorm=True, dropout=False):
        super(HTREncoder, self).__init__()
        
        self.convolutions = nn.Sequential(
        ConvLayer([1, 4, 3], padding=0, stride=2, bn=batchnorm, pool_layer=None),
        ConvLayer([4, 16, 3], padding=0, stride=2, bn=batchnorm, pool_layer=None),
        ConvLayer([16, 32, 3], padding=0, stride=2, bn=batchnorm, pool_layer=None),
        ConvLayer([32, 64, 3], padding=0, stride=1, bn=batchnorm, pool_layer=None))
   
    def forward(self, x):
        h = self.convolutions(x)
        h = F.max_pool2d(h, [h.size(0), 1], padding=[0, 0])
        h = h.squeeze(2)
        return h

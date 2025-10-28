# -*- coding: utf-8 -*-
"""

"""

import time
import math
from functools import partial
from typing import Optional, Callable

import copy
import logging


from os.path import join as pjoin
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage


from os.path import join as pjoin
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms.functional as TF
from einops import rearrange

import timm

from blocks import ChannelAttention, SpatialAttention, DoubleConv, DoubleConv2, Attention_block, CoAttention
from blocks import CoASMamba, CoAMamba, DoubleLCoA


#### Network of MambaCAFU #########################################################  
class MambaCAFU(nn.Module):
    def __init__(self, input_channels=3, num_classes = 1, version = "v2", synapseDB = True):
        super(MambaCAFU, self).__init__()
        print("Start MambaCAFU")
        self.synapseDB=synapseDB
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)       
        
        nb_filter = [32, 64, 128, 256, 512]
        nb_filter2 = [64, 64, 128, 256, 512]
        
        if version =='v0':
            self.transformer = timm.create_model('pvt_v2_b0', pretrained=True, features_only=True)
            pvtdim = [32, 64, 160, 256] 
        elif version =='v2':
            self.transformer = timm.create_model('pvt_v2_b2_li', pretrained=True, features_only=True)
            pvtdim = [64, 128, 320, 512] 
            
        # ResNet
        self.resnet = timm.create_model(
            'resnet18', pretrained=True , features_only=True
        )                    
        
        self.conv0 = DoubleConv(input_channels, nb_filter[0])
        self.CoASMamba1 = CoASMamba(nb_filter[0], pvtdim[0], nb_filter2[0], nb_filter[1])
        self.CoASMamba2 = CoASMamba(nb_filter[1], pvtdim[1], nb_filter2[1], nb_filter[2])
        self.CoASMamba3 = CoASMamba(nb_filter[2], pvtdim[2], nb_filter2[2], nb_filter[3])
        self.CoASMamba4 = CoASMamba(nb_filter[3], pvtdim[3], nb_filter2[3], nb_filter[4])
        
        self.CoAMamba = CoAMamba(nb_filter[3], nb_filter[4], nb_filter[4])
        self.avgpool_x4 = nn.AdaptiveAvgPool2d(14)
        
        self.d4 = DoubleLCoA(nb_filter[3], nb_filter[4], nb_filter[4], nb_filter[3]) 
        self.d3 = DoubleLCoA(nb_filter[2], nb_filter[3], nb_filter[3], nb_filter[2])
        self.d2 = DoubleLCoA(nb_filter[1], nb_filter[2], nb_filter[2], nb_filter[1]) 
        self.d1 = DoubleLCoA(nb_filter[0], nb_filter[1], nb_filter[1], nb_filter[0])           
        
        
        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        # Images
        if self.synapseDB:
            input = input.repeat(1, 3, 1, 1)

        image_size = input.shape
        '''Encoder'''
        # x0 ---> x0
        x0 = self.conv0(input)

        t0, t1, t2, t3 = self.transformer(input)
        r0, r1, r2, r3, r4 = self.resnet(input)
        
        # x1, tr1
        x1 = self.CoASMamba1(self.pool(x0), self.up(t0), r0)
        x2 = self.CoASMamba2(self.pool(x1), self.up(t1), r1) 
        x3 = self.CoASMamba3(self.pool(x2), self.up(t2), r2)
        x4 = self.CoASMamba4(self.pool(x3), self.up(t3), r3)
           
        x5 = self.avgpool_x4(self.CoAMamba(x3, self.up(x4)))

        # Decoder
        d4 = self.d4(x3, self.up(x4), self.up(x5))      
        d3 = self.d3(x2, self.up(x3), self.up(d4))
        d2 = self.d2(x1, self.up(x2), self.up(d3))
        d1 = self.d1(x0, self.up(x1), self.up(d2))
        d0 = self.final(d1)               
        return d0

    
#######################
"""
if __name__ == "__main__":
    # 
    net = MambaCAFU(input_channels=3, num_classes= 9).cuda()

    inp = torch.rand(1,1,224,224).cuda()
    out = net(inp)

    print("out = ",out.shape)
"""


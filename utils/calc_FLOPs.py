from thop import profile
from model import IMDN, IMDN_RTC, SpatialGroupEnhance, IMDModule_speed
import torch
import torch.nn as nn
from fightingcv_attention.attention.ShuffleAttention import ShuffleAttention
from fightingcv_attention.conv.DepthwiseSeparableConvolution import DepthwiseSeparableConvolution
from best import IMDN_DW
from own_model import IMDN_MH
device = torch.device('cpu')
print(device)
width = 256
height = 256
# model = IMDModule_speed(in_channels = 16, distillation_rate=0.25).to(device)
# model = SpatialGroupEnhance(groups = 8).to(device)
# model = ShuffleAttention(channel = 32,G =8).to(device)

model = IMDN_MH().to(device)
img = torch.randn(1,3, width,height).to(device)

flops, params = profile(model, inputs=(img, ), verbose=False)

print('============================')
print(f'FLOPs : { 2*flops/(1e9) } G')
print(f'PARAMS : { params/(1e6) } M ')
print('============================')

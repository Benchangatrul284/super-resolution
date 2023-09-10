import torch
import torch.nn as nn
import sys
import os
from thop import profile
# import your model here
os.chdir(os.path.join(os.getcwd(), 'utils'))
sys.path.append('../')
from model import IMDN_MH
device = torch.device('cpu')
width = 256
height = 256

model = IMDN_MH().to(device)
img = torch.randn(1,3, width,height).to(device)

flops, params = profile(model, inputs=(img, ), verbose=False)

print('============================')
print(f'FLOPs : { 2*flops/(1e9) } G')
print(f'PARAMS : { params/(1e6) } M ')
print('============================')

import torch.nn as nn
from collections import OrderedDict
import torch

class IMDN_MH(nn.Module):
    '''
    IMDN_MH: Multi-Headed version of IMDN
    total 52 channels, 2 heads, 26 channels per head
    '''
    def __init__(self,in_nc=3,nf=52,num_modules=8,num_head=2,out_nc=3,upscale=3):
        super(IMDN_MH,self).__init__()
        self.fea_conv = conv_layer(in_nc, nf, kernel_size=3) # 3 --> 64 with 3*3 conv
        self.split_list = [nf//num_head]*num_head
        self.heads = nn.ModuleList([IMDN_Head(nf=nf//num_head,num_modules=num_modules) for i in range(num_head)])
        self.c = conv_layer(in_channels=nf,out_channels=in_nc,kernel_size=1)
        self.upsampler = pixelshuffle_block(nf, out_nc, upscale_factor=upscale)
        self.cca = CCALayer(nf,reduction=1)
        
    def forward(self,input):
        out_fea = self.fea_conv(input)
        in_list = torch.split(out_fea, self.split_list, dim=1)
        out_list = [self.heads[i](in_list[i]) for i in range(len(self.heads))]
        out_fused_HC = self.cca((torch.cat(out_list,dim=1)))+out_fea
        output = self.upsampler(out_fused_HC)
        return output

    
class IMDN_Head(nn.Module):
    '''
    IMDN_Head: Head module of IMDN_MH
    '''
    def __init__(self,nf,num_modules):
        super(IMDN_Head, self).__init__()
        # IMDBs
        self.IMDB1 = IMDModule_dw(in_channels=nf)
        self.IMDB2 = IMDModule_dw(in_channels=nf)
        self.IMDB3 = IMDModule_dw(in_channels=nf)
        self.IMDB4 = IMDModule_dw(in_channels=nf)
        self.IMDB5 = IMDModule_dw(in_channels=nf)
        self.IMDB6 = IMDModule_dw(in_channels=nf)
        self.IMDB7 = IMDModule_dw(in_channels=nf)
        self.IMDB8 = IMDModule_dw(in_channels=nf)
        self.c = conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')
        self.LR_conv = conv_layer(nf, nf, kernel_size=3)

        
    def forward(self, input):
        out_B1 = self.IMDB1(input)
        out_B2 = self.IMDB2(out_B1)
        out_B3 = self.IMDB3(out_B2)
        out_B4 = self.IMDB4(out_B3)
        out_B5 = self.IMDB5(out_B4)
        out_B6 = self.IMDB6(out_B5)
        out_B7 = self.IMDB7(out_B6)
        out_B8 = self.IMDB8(out_B7)
        # residual link
        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6, out_B7, out_B8], dim=1))
        # out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6], dim=1))
        out_lr = self.LR_conv(out_B)+input
        return out_lr

    

class IMDModule_dw(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super(IMDModule_dw, self).__init__()
        self.distilled_channels = int(in_channels * distillation_rate)
        self.remaining_channels = int(in_channels - self.distilled_channels)
        self.c1 = conv_layer_dw(in_channels, in_channels, 3)
        self.c2 = conv_layer_dw(self.remaining_channels, in_channels, 3)
        self.c3 = conv_layer_dw(self.remaining_channels, in_channels, 3)
        self.c4 = conv_layer_dw(self.remaining_channels, self.distilled_channels, 3)
        self.act = activation('lrelu', neg_slope=0.05)
        self.c5 = conv_layer_dw(self.distilled_channels * 4, in_channels, 1)
        self.cca = CCALayer(self.distilled_channels * 4,reduction=1)
        
    def forward(self, input):
        '''
        distilled_channels performs residual link
        remaining_channels performs convolution
        '''
        out_c1 = self.act(self.c1(input))
        distilled_c1, remaining_c1 = torch.split(out_c1, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c2 = self.act(self.c2(remaining_c1))
        distilled_c2, remaining_c2 = torch.split(out_c2, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c3 = self.act(self.c3(remaining_c2))
        distilled_c3, remaining_c3 = torch.split(out_c3, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c4 = self.c4(remaining_c3)

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, out_c4], dim=1)
        out_fused = self.c5(self.cca(out)) + input
        return out_fused
    
    
def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias, dilation=dilation,
                     groups=groups)

def conv_layer_dw(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding=padding, bias=bias, dilation=dilation,
                    groups=in_channels),
        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=bias, dilation=1, groups=groups)
    )


def norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
               pad_type='zero', norm_type=None, act_type='relu'):
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
                  dilation=dilation, bias=bias, groups=groups)
    a = activation(act_type) if act_type else None
    n = norm(norm_type, out_nc) if norm_type else None
    return sequential(p, c, n, a)


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


class ShortcutBlock(nn.Module):
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))

def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)

def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


# contrast-aware channel attention module
class CCALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CCALayer, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )


    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
    
def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv1 = conv_layer_dw(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv1,pixel_shuffle)
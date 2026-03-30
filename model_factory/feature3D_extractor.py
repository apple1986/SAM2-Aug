# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

# Modified from
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
# Copyright 2020 Ross Wightman, Apache-2.0 License
## aping: use 3x3 and 1x1 depthwise-conv
from turtle import forward
from sympy import prem
import torch
import itertools
import torch.nn as nn
import torch.nn.functional as F



#################
## Attention
class Linear_BN(torch.nn.Sequential):
    def __init__(self, a, b, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Linear(a, b, bias=False))
        bn = torch.nn.BatchNorm1d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)


    @torch.no_grad()
    def fuse(self):
        l, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = l.weight * w[:, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

    def forward(self, x):
        l, bn = self._modules.values()
        x = l(x)
        return bn(x.flatten(0, 1)).reshape_as(x)


class Residual(torch.nn.Module):
    def __init__(self, m, drop):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)


class Attention(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 activation=None,
                 resolution=(14,14,14)):
        super().__init__()
        self.num_heads = num_heads # number of heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim # dimenison of Key
        self.nh_kd = nh_kd = key_dim * num_heads # total dimension of key: head_num * each_key_dim
        self.d = int(attn_ratio * key_dim) 
        self.dh = int(attn_ratio * key_dim) * num_heads
        h = self.dh + nh_kd * 2 # total dimension of key, qury, value
        self.qkv = Linear_BN(dim, h)
        self.proj = torch.nn.Sequential(activation(), Linear_BN(
            self.dh, dim, bn_weight_init=0,))

        points = list(itertools.product(range(resolution[0]), range(resolution[1]), range(resolution[2])))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                # Compute 3D offset as (ΔD, ΔH, ΔW)
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]), abs(p1[2] - p2[2]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        # Define attention biases
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets))
        )

        # Register the buffer for attention bias indices
        self.register_buffer(
            'attention_bias_idxs',
            torch.LongTensor(idxs).view(N, N)
        )

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):  # x (B,N,C)
        B, N, C = x.shape
        qkv = self.qkv(x) # change channel numbers
        q, k, v = qkv.view(B, N, self.num_heads, -
                           1).split([self.key_dim, self.key_dim, self.d], dim=3)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = (
            (q @ k.transpose(-2, -1)) * self.scale
            +
            (self.attention_biases[:, self.attention_bias_idxs]
             if self.training else self.ab)
        )
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.dh)
        x = self.proj(x)
        return x


class Stage_Att(torch.nn.Module):
    def __init__(self, dim=64, key_dim=16, num_heads=4, activation=nn.Hardswish, resolution=(4, 4, 4), drop_rate=0):
        super().__init__()
        self.act = activation
        # attention lay
        self.att1 = Residual(
                            Attention(dim=dim, key_dim=key_dim, num_heads=num_heads, activation=self.act, resolution=resolution),
                            drop=drop_rate)

        ## mlp
        self.mlp1 = Residual(torch.nn.Sequential(
                            Linear_BN(dim, dim*2, ),
                            self.act(),
                            Linear_BN(dim*2, dim, bn_weight_init=0,
                                    ),
                        ), drop_rate)
        self.trans1 = nn.Sequential(self.att1, self.mlp1)

        # attention lay
        self.att2 = Residual(
                            Attention(dim=dim, key_dim=key_dim, num_heads=num_heads, activation=self.act, resolution=resolution),
                            drop=drop_rate)

        ## mlp
        self.mlp2 = Residual(torch.nn.Sequential(
                            Linear_BN(dim, dim*2),
                            self.act(),
                            Linear_BN(dim*2, dim, bn_weight_init=0,
                                    ),
                        ), drop_rate)
        self.trans2 = nn.Sequential(self.att2, self.mlp2)

    def forward(self, x):
        x = self.trans1(x)
        x = self.trans2(x)

        return x
################################
## build efficient encoder
class Conv3d_BN_Eff_Res_en(nn.Module):
    def __init__(self, in_ch, out_ch, ks=1, stride=1, pad=1, act=nn.Hardswish):
        super().__init__()
        self.conv_bn = nn.Sequential(nn.Conv3d(in_ch, in_ch, ks, stride, pad, groups=in_ch),
                                            nn.BatchNorm3d(in_ch),
                                            act(),
                                            nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=1),
                                            nn.BatchNorm3d(out_ch),
                                            )
        self.skip = nn.Sequential(
            nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=1),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.BatchNorm3d(out_ch),
        )

    def forward(self, x):
        feat = self.conv_bn(x)
        feat = feat + self.skip(x)

        return feat
    
## add dilated conv for encoder
class Conv3d_BN_Eff_Res_Di_En(nn.Module):
    def __init__(self, in_ch, out_ch, ks=3, stride=1, pad=1, act=nn.Hardswish):
        super().__init__()

        self.dconv1 = nn.Conv3d(in_ch, in_ch, ks, stride, padding=pad, dilation=1, groups=in_ch) # rate=1
        self.dconv2 = nn.Conv3d(in_ch, in_ch, ks, stride, padding=2, dilation=2, groups=in_ch) # rate=2
        self.dconv3 = nn.Conv3d(in_ch, in_ch, ks, stride, padding=3, dilation=3, groups=in_ch) # rate=3
        self.dconv4 = nn.Conv3d(in_ch, in_ch, ks, stride, padding=4, dilation=4, groups=in_ch) # rate=4

        self.conv1x1= nn.Sequential(
            nn.BatchNorm3d(in_ch),
            act(),
            nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=1),
            nn.BatchNorm3d(out_ch),
            )
        
        self.skip = nn.Sequential(
            nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=1),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.BatchNorm3d(out_ch),
            )

    def forward(self, x):
        feat1 = self.dconv1(x)
        feat2 = self.dconv2(x)
        feat3 = self.dconv3(x)
        feat4 = self.dconv4(x)
        feat = feat1 + feat2 + feat3 + feat4 # multi-scale fusion
        feat = self.conv1x1(feat)
        feat = feat + self.skip(x)

        return feat

class Conv3d_BN_Eff_Res_Di_En_Re(nn.Module):
    """
    Did not do maxpooling in axial direction
    """
    def __init__(self, in_ch, out_ch, ks=3, stride=1, pad=1, act=nn.Hardswish):
        super().__init__()

        self.dconv1 = nn.Conv3d(in_ch, in_ch, ks, stride, padding=pad, dilation=1, groups=in_ch) # rate=1
        self.dconv2 = nn.Conv3d(in_ch, in_ch, ks, stride, padding=2, dilation=2, groups=in_ch) # rate=2
        self.dconv3 = nn.Conv3d(in_ch, in_ch, ks, stride, padding=3, dilation=3, groups=in_ch) # rate=3
        self.dconv4 = nn.Conv3d(in_ch, in_ch, ks, stride, padding=4, dilation=4, groups=in_ch) # rate=4

        self.conv1x1= nn.Sequential(
            nn.BatchNorm3d(in_ch),
            act(),
            nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=1),
            nn.BatchNorm3d(out_ch),
            )
        
        self.skip = nn.Sequential(
            nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=(2,2,1)),
            nn.BatchNorm3d(out_ch),
            )

    def forward(self, x):
        feat1 = self.dconv1(x)
        feat2 = self.dconv2(x)
        feat3 = self.dconv3(x)
        feat4 = self.dconv4(x)
        feat = feat1 + feat2 + feat3 + feat4 # multi-scale fusion
        feat = self.conv1x1(feat)
        feat = feat + self.skip(x)

        return feat

class Conv3d_BN_Eff_Res_Di_En_Focus(nn.Module):
    """
    Did not do maxpooling in axial direction
    """
    def __init__(self, in_ch, out_ch, ks=3, stride=1, pad=1, act=nn.Hardswish):
        super().__init__()

        self.dconv1 = nn.Conv3d(in_ch, in_ch, ks, stride, padding=pad, dilation=1, groups=in_ch) # rate=1
        self.dconv2 = nn.Conv3d(in_ch, in_ch, ks, stride, padding=2, dilation=2, groups=in_ch) # rate=2
        self.dconv3 = nn.Conv3d(in_ch, in_ch, ks, stride, padding=3, dilation=3, groups=in_ch) # rate=3
        self.dconv4 = nn.Conv3d(in_ch, in_ch, ks, stride, padding=4, dilation=4, groups=in_ch) # rate=4

        self.conv1x1= nn.Sequential(
            nn.BatchNorm3d(in_ch),
            act(),
            nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=1),
            nn.BatchNorm3d(out_ch),
            )
        
        self.skip = nn.Sequential(
            nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=stride),
            nn.BatchNorm3d(out_ch),
            )

    def forward(self, x):
        feat1 = self.dconv1(x)
        feat2 = self.dconv2(x)
        feat3 = self.dconv3(x)
        feat4 = self.dconv4(x)
        feat = feat1 + feat2 + feat3 + feat4 # multi-scale fusion
        feat = self.conv1x1(feat)
        feat = feat + self.skip(x)

        return feat

class Conv2d_BN_Eff_Res_en(nn.Module):
    def __init__(self, in_ch, out_ch, ks=1, stride=1, pad=1, act=nn.Hardswish):
        super().__init__()
        self.conv_bn = nn.Sequential(nn.Conv2d(in_ch, in_ch, ks, stride, pad, groups=in_ch),
                                            nn.BatchNorm2d(in_ch),
                                            act(),
                                            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=1),
                                            nn.BatchNorm2d(out_ch),
                                            )
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        feat = self.conv_bn(x)
        feat = feat + self.skip(x)

        return feat

##################################
##################################

## add dilated conv in the first layer
class Feature_DiFPN(torch.nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, in_channels=3, ## aping: should change
                 out_channels=2,
                 img_size = (512, 512, 128), # 128x128x128
                 n = 128 ## cnn num channel
          ):
        super().__init__()

        # self.out_channels = out_channels
        activation = torch.nn.Hardswish
        self.cnn_b1 = torch.nn.Sequential(
            Conv3d_BN_Eff_Res_Di_En(in_channels, 2, 3, 2, 1,), activation())
        self.cnn_b2 = torch.nn.Sequential(
            Conv2d_BN_Eff_Res_en(128, 256, 3, 2, 1, ), activation())

    def forward(self, x):
        B, _, H, W, D = x.shape
        x_cnn_1 = self.cnn_b1(x) # Bx8x256x256x64
        x_cnn_1 = x_cnn_1.permute(0, 1, 4, 2, 3).contiguous().reshape(B, -1, 256, 256) # Bx(2*64)x256x256
        x_cnn_2 = self.cnn_b2(x_cnn_1) # Bx256x128x128

        return [x_cnn_1, x_cnn_2]

## add dilated conv in the first layer
## reduce the slice number
class Feature_DiFPN_Re(torch.nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, in_channels=3, ## aping: should change
                 out_channels=2,
                 img_size = (512, 512, 8), # input the slice near the bbox
                 n = 128 ## cnn num channel
          ):
        super().__init__()

        # self.out_channels = out_channels
        activation = torch.nn.Hardswish
        self.cnn_b1 = torch.nn.Sequential(
            Conv3d_BN_Eff_Res_Di_En_Re(in_channels, 2, 3, (2,2,1), 1,), activation())
        self.cnn_b2 = torch.nn.Sequential(
            Conv2d_BN_Eff_Res_en(16, 32, 3, 2, 1, ), activation())


    def forward(self, x):
        B, _, H, W, D = x.shape
        x_cnn_1 = self.cnn_b1(x) # Bx8x256x256x64
        x_cnn_1 = x_cnn_1.permute(0, 1, 4, 2, 3).contiguous().reshape(B, -1, 256, 256) # Bx(2*8)x256x256
        x_cnn_2 = self.cnn_b2(x_cnn_1) # Bx32x128x128

        return [x_cnn_1, x_cnn_2]


## add dilated conv in the first layer
## reduce the slice number
# focus on the center of lesion
class Feature_DiFPN_Focus(torch.nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, in_channels=3, ## aping: should change
                 out_channels=2,
                 img_size = (256, 256, 8), # focus on the lesion
                 n = 128 ## cnn num channel
          ):
        super().__init__()

        # self.out_channels = out_channels
        activation = torch.nn.ReLU
        self.cnn_b1 = torch.nn.Sequential(
            Conv3d_BN_Eff_Res_Di_En_Focus(in_channels, 2, 3, (1,1,1), 1,), activation())
        self.cnn_b2 = torch.nn.Sequential(
            Conv2d_BN_Eff_Res_en(16, 32, 3, 2, 1, ), activation())


    def forward(self, x):
        B, _, H, W, D = x.shape
        x_cnn_1 = self.cnn_b1(x) # Bx8x256x256x64
        x_cnn_1 = x_cnn_1.permute(0, 1, 4, 2, 3).contiguous().reshape(B, -1, 256, 256) # Bx(2*8)x256x256
        x_cnn_2 = self.cnn_b2(x_cnn_1) # Bx32x128x128

        return [x_cnn_1, x_cnn_2]

#################################
def netParams(model):
    """
    computing total network parameters
    args:
       model: model
    return: the number of parameters
    """
    total_paramters = 0
    for parameter in model.parameters():
        i = len(parameter.size())
        p = 1
        for j in range(i):
            p *= parameter.size(j)
        total_paramters += p

    return total_paramters


if __name__ == '__main__':
    NUM_D = 8 # 128
    x = torch.randn((3, 3, 256, 256, NUM_D)) # BCHWD 96,256,256 256, 256, 256
    # x_pre = torch.randn(2, 2, 512, 512, NUM_D)

    net = Feature_DiFPN_Focus(img_size=(256, 256, NUM_D), n=NUM_D, in_channels=3)
    x = net(x)
    print(x[0].shape)
    print(x[1].shape)
    total_paramters = netParams(net)
    print("the number of parameters: %d ==> %.2f M" % (total_paramters, (total_paramters / 1e6)))


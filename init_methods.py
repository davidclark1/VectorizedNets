import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from local2d import lc_forward, lc_backward, lc_compute_grads


"""
Linear inits
"""

def init_linear(weight, first_layer=False, mono=False):
    if mono:
        return init_linear_mono(weight, first_layer)
    out_features, in_features = weight.shape
    weight[:] = 0.
    f = 1. if first_layer else 0.5
    weight.normal_(0., 1./np.sqrt(f * in_features))

def init_linear_mono(weight, first_layer):
    if first_layer:
        return init_linear_mono_l0(weight)
    out_features, in_features = weight.shape
    if ((out_features % 2) != 0 and out_features != 1) or ((in_features % 2) != 0):
        raise ValueError("out_features and in_features must both be even (out_features = 1 is okay, too)")
    weight[:] = 0.
    W = torch.randn(max(out_features//2, 1), in_features//2, device=weight.device) / np.sqrt(0.25 * in_features)
    #+ outputs 
    weight[::2, ::2] = F.relu(W)
    weight[::2, 1::2] = F.relu(-W)
    if out_features > 1:
        #- outputs
        weight[1::2, ::2] = F.relu(-W)
        weight[1::2, 1::2] = F.relu(W)

def init_linear_mono_l0(weight):
    out_features, in_features = weight.shape
    if ((out_features % 2) != 0) or ((in_features % 2) != 0):
        raise ValueError("out_features and in_features must both be even")
    weight[:] = 0.
    W = torch.randn(out_features//2, in_features, device=weight.device) / np.sqrt(in_features)
    weight[::2] = W
    weight[1::2] = -W

"""
Convolutional inits
"""

def init_conv(weight, first_layer=False, mono=False):
    if mono:
        return init_conv_mono(weight, first_layer)
    weight[:] = 0.
    out_channels, in_channels, kernel_size = weight.shape[:3] #assumes square kernel
    f = 1 if first_layer else 0.5
    weight.normal_(0., 1./np.sqrt(f * in_channels * kernel_size**2))

def init_conv_mono(weight, first_layer):
    if first_layer:
        return init_conv_mono_l0(weight)
    out_channels, in_channels, kernel_size = weight.shape[:3] #assumes square kernel
    if ((out_channels % 2) != 0) or ((in_channels % 2) != 0):
        raise ValueError("out_channel and in_channels must both be even")
    weight[:] = 0.
    W = torch.randn(out_channels//2, in_channels//2, kernel_size, kernel_size, device=weight.device) / np.sqrt(0.25 * in_channels * kernel_size**2)
    #+ outputs
    weight[::2, ::2] = F.relu(W)
    weight[::2, 1::2] = F.relu(-W)
    #- outputs
    weight[1::2, ::2] = F.relu(-W)
    weight[1::2, 1::2] = F.relu(W)

def init_conv_mono_l0(weight):
    out_channels, in_channels, kernel_size = weight.shape[:3] #assumes square kernel
    if ((out_channels % 2) != 0) or ((in_channels % 2) != 0):
        raise ValueError("out_channel and in_channels must both be even")
    weight[:] = 0.
    filter_shape_3d = weight.shape[1:]
    W = torch.randn((out_channels//2,) + filter_shape_3d, device=weight.device) / np.sqrt(in_channels * kernel_size**2)
    weight[::2] = W
    weight[1::2] = -W

"""
Locally connected inits
"""
    
def init_local(weight, first_layer=False, mono=False):
    if mono:
        return init_local_mono(weight, first_layer)
    weight[:] = 0.
    out_channels, h_out, w_out, in_channels, kernel_size = weight.shape[:5]
    f = 1 if first_layer else 0.5
    weight.normal_(0., 1./np.sqrt(f * in_channels * kernel_size**2))

def init_local_mono(weight, first_layer):
    if first_layer:
        return init_local_mono_l0(weight)
    out_channels, h_out, w_out, in_channels, kernel_size = weight.shape[:5]
    if ((out_channels % 2) != 0) or ((in_channels % 2) != 0):
        raise ValueError("out_channel and in_channels must both be even")
    weight[:] = 0.
    W = torch.randn(out_channels//2, h_out, w_out, in_channels//2, kernel_size, kernel_size, device=weight.device) / np.sqrt(0.25 * in_channels * kernel_size**2)
    weight[::2, :, :, ::2] = F.relu(W)
    weight[::2, :, :, 1::2] = F.relu(-W)
    weight[1::2, :, :, 1::2] = F.relu(W)
    weight[1::2, :, :, ::2] = F.relu(-W)

def init_local_mono_l0(weight):
    out_channels, h_out, w_out, in_channels, kernel_size = weight.shape[:5]
    if ((out_channels % 2) != 0) or ((in_channels % 2) != 0):
        raise ValueError("out_channel and in_channels must both be even")
    weight[:] = 0.
    filter_shape_3d = weight.shape[3:]
    W = torch.randn((out_channels//2, h_out, w_out,) + filter_shape_3d, device=weight.device) / np.sqrt(in_channels * kernel_size**2)
    weight[::2] = W
    weight[1::2] = -W
    



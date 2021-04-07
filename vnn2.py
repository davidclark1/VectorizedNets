import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from local2d import lc_forward, lc_backward, lc_compute_grads

"""
Input expansion utiltiies
"""

def expand_input_conv(input, category_dim):
    #input = (batch, channels, width, height)
    #output = (batch, K, K*channels, width, height)
    batch_size, in_channels = input.shape[:2]
    expanded_input = torch.zeros((batch_size, category_dim, in_channels*category_dim) + input.shape[2:], device=input.device)
    for i in range(category_dim):
        expanded_input[:, i, i*in_channels:(i+1)*in_channels] = input
    return expanded_input

def expand_input(input, category_dim):
    batch_dim, input_dim = input.shape
    expanded_input = torch.zeros(batch_dim, category_dim, category_dim*input_dim, device=input.device)
    for i in range(category_dim):
        expanded_input[:, i, i*input_dim:(i+1)*input_dim] = input
    return expanded_input

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
    

"""
Vectorized linear layer
"""

class Linear(nn.Module):
    def __init__(self, category_dim, in_features, out_features, first_layer=False, mono=False):
        super().__init__()
        self.category_dim = category_dim
        self.in_features = in_features
        self.out_features = out_features
        self.first_layer = first_layer
        self.mono = mono
        self.weight = nn.Parameter(torch.zeros(out_features, in_features), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(category_dim, out_features), requires_grad=False)
        init_linear(self.weight, first_layer=first_layer, mono=mono)
        if first_layer:
            self.weight *= np.sqrt(category_dim)

    def forward(self, input):
        self.input = input.detach()
        output = torch.matmul(input, self.weight.T) + self.bias
        return output

    def custom_backward(self, grad_output, output_error):
        self.set_grad(grad_output, output_error)
        grad_input = torch.matmul(grad_output, self.weight)
        return grad_input

    def set_grad(self, grad_output, output_error):
        #n = input feature dim, m = output feature dim
        dot_prods = torch.einsum("bkn,bk->bn", self.input, output_error)
        grad_weight = torch.einsum("bm,bn->mn", grad_output, dot_prods)
        grad_bias = torch.einsum("bk,bm->km", output_error, grad_output)
        set_or_add_grad(self.weight, grad_weight)
        set_or_add_grad(self.bias, grad_bias)

    def post_step_callback(self):
        if self.mono and (not self.first_layer):
            self.weight.clamp_(min=0)
                
"""
Vectorized 2d convolutional layer
"""

def convolutional_outer_product(x1, x2, kernel_size, stride=1, padding=0):
    #x1 = (batch_size, in_channels, h_in, w_in)
    #x2 = (batch_size, out_channels, h_out, w_out)
    #out = (batch_size, out_chanels, in_channels, kernel_size, kernel_size)
    batch_size = x1.shape[0]
    if x2.shape[0] != batch_size:
        raise ValueError("x1 and x2 must have the same batch size")
    in_channels = x1.shape[1]
    out_channels = x2.shape[1]
    h_in, w_in = x1.shape[2:]
    h_out, w_out = x2.shape[2:]
    x1_prime = x1.permute(1, 0, 2, 3)
    x2_prime = x2.reshape(batch_size*out_channels, h_out, w_out).unsqueeze(1)
    out_prime = F.conv2d(input=x1_prime, weight=x2_prime, groups=batch_size, dilation=stride, stride=1, padding=padding)
    out_prime = out_prime[:, :, :kernel_size, :kernel_size]
    out = out_prime.view(in_channels, batch_size, out_channels, kernel_size, kernel_size).permute(1, 2, 0, 3, 4)
    return out

class Conv2d(nn.Module):
    def __init__(self, category_dim, in_channels, out_channels, kernel_size, stride=1, padding=0, mono=False, first_layer=False):
            super().__init__()
            self.category_dim = category_dim
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernel_size = kernel_size
            self.stride = stride
            self.padding = padding
            self.mono = mono
            self.first_layer = first_layer
            self.weight = nn.Parameter(torch.zeros(out_channels, in_channels, kernel_size, kernel_size), requires_grad=False)
            self.bias = nn.Parameter(torch.zeros(category_dim, out_channels), requires_grad=False)
            init_conv(self.weight, first_layer=first_layer, mono=mono)
            if first_layer:
                self.weight *= np.sqrt(category_dim)
    
    def forward(self, input):
        #input = (batch_dim, category_dim, channels, width, height)
        self.input = input.detach()
        batch_size, category_dim = input.shape[:2]
        CWH = input.shape[2:]
        input_reshaped = input.view((batch_size*category_dim,) + CWH)
        output_reshaped = F.conv2d(input_reshaped, weight=self.weight, stride=self.stride, padding=self.padding)
        output = output_reshaped.view((batch_size, category_dim) + output_reshaped.shape[1:])
        output = output + self.bias[None, :, :, None, None]
        return output

    def custom_backward(self, grad_output, output_error):
        self.set_grad(grad_output, output_error)
        #TODO: compute what output_padding should be so that works for all strides
        grad_input = F.conv_transpose2d(grad_output, weight=self.weight, stride=self.stride, padding=self.padding)
        if grad_input.shape != self.input.shape[0:1] + self.input.shape[2:]:
            raise ValueError("Shape mismatch in backwards pass of Conv2d")
        return grad_input

    def set_grad(self, grad_output, output_error):
        #grad_output = (batch_size, out_channels, h_out, w_out)
        #output_error = (batch_size, category_dim)
        x1 = torch.einsum("bkchw,bk->bchw", self.input, output_error)
        x2 = grad_output
        outer = convolutional_outer_product(x1, x2, self.kernel_size, stride=self.stride, padding=self.padding)
        grad_weight = outer.sum(dim=0)
        grad_bias = torch.einsum("bc,bk->bkc", grad_output.sum(dim=(2, 3)), output_error).sum(dim=0)
        set_or_add_grad(self.weight, grad_weight)
        set_or_add_grad(self.bias, grad_bias)

    def post_step_callback(self):
        if self.mono and (not self.first_layer):
            self.weight.clamp_(min=0)
                
"""
Vectorized locally connected layer
"""

class VecLocal2d(nn.Module):
    def __init__(self, category_dim, in_channels, out_channels, kernel_size, h_in, w_in, stride=1, padding=0, mono=False, first_layer=False):
            super().__init__()
            self.category_dim = category_dim
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernel_size = kernel_size
            self.h_in = h_in
            self.w_in = w_in
            self.stride = stride
            self.padding = padding
            h_out = int(np.floor(((h_in + 2*padding - kernel_size)/stride) + 1))
            w_out = int(np.floor(((w_in + 2*padding - kernel_size)/stride) + 1))
            self.h_out = h_out
            self.w_out = w_out
            self.first_layer = first_layer
            self.mono = mono
            self.weight = nn.Parameter(torch.zeros(out_channels, h_out, w_out, in_channels, kernel_size, kernel_size), requires_grad=False)
            self.bias = nn.Parameter(torch.zeros(out_channels, h_out, w_out), requires_grad=False)
            init_local(self.weight, first_layer=first_layer, mono=mono)
            if first_layer:
                self.weight *= np.sqrt(category_dim)

    def forward(self, input):
        #input = (batch_dim, category_dim, channels, width, height)
        self.input = input.detach()
        batch_size, category_dim = input.shape[:2]
        CWH = input.shape[2:]
        input_reshaped = input.view((batch_size*category_dim,) + CWH)
        output_reshaped = lc_forward(input_reshaped, weight=self.weight, bias=None, stride=self.stride, padding=self.padding)
        output = output_reshaped.view((batch_size, category_dim) + output_reshaped.shape[1:]) + self.bias
        return output

    def custom_backward(self, grad_output, output_error):
        input_dp = torch.einsum("bkchw,bk->bchw", self.input, output_error)
        grad_weight, grad_bias = lc_compute_grads(input_dp, grad_output, self.kernel_size, True, self.stride, self.padding)
        set_or_add_grad(self.weight, grad_weight)
        set_or_add_grad(self.bias, grad_bias)
        grad_input = lc_backward(grad_output, weight=self.weight, input_shape=self.input.shape[2:], stride=self.stride, padding=self.padding)
        return grad_input
    
    def post_step_callback(self):
        if self.mono and (not self.first_layer):
            with torch.no_grad():
                self.conv.weight.clamp_(min=0)


"""
Vectorized nonlinearities
"""

class tReLU(nn.Module):
    def __init__(self, category_dim, num_features):
        super().__init__()
        t = torch.zeros(category_dim, num_features)
        t_half = torch.randint(0, 2, (category_dim, num_features//2)).float()*2 - 1
        t[:, ::2] = t_half
        t[:, 1::2] = -t_half
        self.t = nn.Parameter(t, requires_grad=False)

    def forward(self, input):
        mask = ((input.detach() * self.t[None]).sum(dim=1) >= 0.).float()
        self.mask = mask
        output = input * mask[:, None]
        return output

    def custom_backward(self, grad_output, output_error):
        grad_input = grad_output * self.mask
        return grad_input
    
class ctReLU(nn.Module):
    def __init__(self, category_dim, num_channels, height, width, share_t=True):
        super().__init__()
        self.share_t = share_t
        t = torch.zeros(category_dim, num_channels, height, width)
        if share_t:
            t_half = torch.randint(0, 2, (category_dim, num_channels//2)).float()*2 - 1 #NOTE: same t vec per channel
            t[:, ::2, :, :] = t_half[:, :, None, None]
            t[:, 1::2, :, :] = -t_half[:, :, None, None]
        else:
            t_half = torch.randint(0, 2, (category_dim, num_channels//2, height, width)).float()*2 - 1
            t[:, ::2, :, :] = t_half
            t[:, 1::2, :, :] = -t_half
        self.t = nn.Parameter(t, requires_grad=False)

    def forward(self, input):
        mask = ((input.detach() * self.t[None]).sum(dim=1) >= 0.).float()
        self.mask = mask
        output = input * mask[:, None]
        return output

    def custom_backward(self, grad_output, output_error):
        grad_input = grad_output * self.mask
        return grad_input
    
"""
Misc. vectorized layers
"""

class Flatten(nn.Module):
    def forward(self, input):
        #print("Here")
        #(batch, cat, channels, width, height)
        self.input_shape = input.shape
        input_reshaped = input.permute(0, 1, 3, 4, 2).reshape(input.shape[:2] + (np.prod(input.shape[2:]),))
        return input_reshaped

    def custom_backward(self, grad_output, output_error):
        #grad_output = (batch, channels*width*height)
        b, cat, c, h, w = self.input_shape
        grad_input = grad_output.view(b, h, w, c).permute(0, 3, 1, 2) #.clone()
        return grad_input


class AvgPool2d(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.pool = nn.AvgPool2d(kernel_size)

    def forward(self, input):
        h, w = input.shape[-2:]
        if (h % self.kernel_size != 0) or (w % self.kernel_size != 0):
            raise ValueError("kernel_size must evenly divide input width/height")
        batch_size, category_dim = input.shape[:2]
        CWH = input.shape[2:]
        input_reshaped = input.view((batch_size*category_dim,) + CWH)
        output_reshaped = self.pool(input_reshaped)
        output = output_reshaped.view((batch_size, category_dim) + output_reshaped.shape[1:])
        return output

    def custom_backward(self, grad_output, output_error):
        #grad_output = (batch, c, h, w)
        grad_input = (torch.repeat_interleave(torch.repeat_interleave(grad_output, self.kernel_size, dim=2), self.kernel_size, dim=3) / self.kernel_size**2)
        return grad_input


"""
Utilities for filling .grad attributes with the bio-plausible learning updates
"""

def set_or_add_grad(param, grad_val):
    with torch.no_grad():
        if param.grad is None:
            param.grad = grad_val.detach() #probably don't need detach
        else:
            param.grad += grad_val.detach() #again, probably don't need detach

def set_model_grads(model, output, labels, learning_rule="bp", reduction="mean"):
    if len(output.shape) != 2:
        raise ValueError("output.shape must be (batch_size, category_dim)")
    targets = torch.eye(output.shape[1], device=output.device)[labels]
    output_error = F.softmax(output.detach(), dim=1) - targets
    batch_size = output.shape[0]
    g = torch.ones(batch_size, 1, device=output.device)
    if reduction == "mean":
        g /= batch_size
    for i in list(range(len(model)))[::-1]:
        layer = model[i]
        g = layer.custom_backward(g, output_error)


"""
def set_model_grads(model, output, labels):
    if len(output.shape) != 2:
        raise ValueError("output.shape must be (batches, categories)")
    targets = torch.eye(output.shape[1], device=output.device)[labels.detach()]
    output_error = F.softmax(output.detach(), dim=1) - targets
    n1, n2 = 0, 0
    for i in range(len(model)):
        layer = model[i]
        if layer.__class__.__name__ in ('Conv2d', 'Linear', 'VecLocal2d'):
            if (i < len(model) - 1) and (model[i + 1].__class__.__name__ in ('tReLU', 'ctReLU')):
                mask = model[i + 1].mask
                n1 += 1
            else:
                mask = torch.ones(layer.mask_shape, device=output.device)
                n2 += 1
            layer.set_grad(mask, output_error)
    return (n1, n2)
"""

def post_step_callback(model):
    for module in model:
        if hasattr(module, "post_step_callback"):
            module.post_step_callback()


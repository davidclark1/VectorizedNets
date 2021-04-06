import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Input expansion utiltiies
"""

def expand_input_conv(input, category_dim):
    #input = (batch, channels, width, height)
    #output = (batch, K, K*channels, width, height)
    batch_size, in_channels = input.shape[:2]
    expanded_input = torch.zeros((batch_size, category_dim, in_channels*category_dim) + input.shape[2:],
        device=input.device)
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
    h_out, w_out, out_channels, in_channels, kernel_size = weight.shape[:5]
    f = 1 if first_layer else 0.5
    weight.normal_(0., 1./np.sqrt(f * in_channels * kernel_size**2))

def init_local_mono(weight, first_layer):
    if first_layer:
        return init_local_mono_l0(weight)
    h_out, w_out, out_channels, in_channels, kernel_size = weight.shape[:5]
    if ((out_channels % 2) != 0) or ((in_channels % 2) != 0):
        raise ValueError("out_channel and in_channels must both be even")
    weight[:] = 0.
    W = torch.randn(h_out, w_out, out_channels//2, in_channels//2, kernel_size, kernel_size, device=weight.device) / np.sqrt(0.25 * in_channels * kernel_size**2)
    weight[:, :, ::2, ::2] = F.relu(W)
    weight[:, :, ::2, 1::2] = F.relu(-W)
    weight[:, :, 1::2, 1::2] = F.relu(W)
    weight[:, :, 1::2, ::2] = F.relu(-W)

def init_local_mono_l0(weight):
    h_out, w_out, out_channels, in_channels, kernel_size = weight.shape[:5]
    if ((out_channels % 2) != 0) or ((in_channels % 2) != 0):
        raise ValueError("out_channel and in_channels must both be even")
    weight[:] = 0.
    filter_shape_3d = weight.shape[3:]
    W = torch.randn((h_out, w_out, out_channels//2,) + filter_shape_3d, device=weight.device) / np.sqrt(in_channels * kernel_size**2)
    weight[:, :, ::2] = W
    weight[:, :, 1::2] = -W
    

"""
Vectorized linear layer
"""

class Linear(nn.Module):
    def __init__(self, category_dim, in_features, out_features, first_layer=False, mono=False, device="cpu"):
        super().__init__()
        self.category_dim = category_dim
        self.in_features = in_features
        self.out_features = out_features
        self.mono = mono
        self.first_layer = first_layer
        self.mono = mono
        self.device = device
        self.weight = nn.Parameter(torch.zeros(out_features, in_features, device=device), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(category_dim, out_features, device=device), requires_grad=False)
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

    def post_step_callback(self):
        if self.mono and (not self.first_layer):
            self.weight.clamp_(min=0)

    def set_grad(self, grad_output, output_error):
        #i = batch dim
        #j = category dim
        #n = input feature dim
        #m = output feature dim
        dot_prods = torch.einsum("ijn,ij->in", self.input, output_error)
        delta_W = torch.einsum("im,in->mn", grad_output, dot_prods) / len(self.input)
        delta_b = torch.einsum("ij,im->jm", output_error, grad_output) / len(self.input)
        set_or_add_grad(self.weight, delta_W)
        set_or_add_grad(self.bias, delta_b)
                
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
    def __init__(self, category_dim, in_channels, out_channels, kernel_size, stride=1, padding=0, mono=False, first_layer=False, device="cpu"):
            super().__init__()
            self.category_dim = category_dim
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernel_size = kernel_size
            self.stride = stride
            self.padding = padding
            self.mono = mono
            self.first_layer = first_layer
            self.device = device
            self.weight = nn.Parameter(torch.zeros(out_channels, in_channels, kernel_size, kernel_size), requires_grad=False)
            self.bias = nn.Parameter(torch.zeros(category_dim, out_channels, device=device), requires_grad=False)
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
        grad_input = F.conv_transpose2d(grad_output, weight=self.weight, stride=self.stride, padding=self.padding) #output padding???
        if grad_input.shape != self.input.shape[0:1] + self.input.shape[2:]:
            raise ValueError("Shape mismatch in backwards pass of Conv2d")
        return grad_input
    
    def post_step_callback(self):
        if self.mono and (not self.first_layer):
            self.weight.clamp_(min=0)

    def set_grad(self, grad_output, output_error):
        #grad_output = (batch_size, out_channels, h_out, w_out)
        #output_error = (batch_size, category_dim)
        x1 = torch.einsum("bkchw,bk->bchw", self.input, output_error)
        x2 = grad_output
        outer = convolutional_outer_product(x1, x2, self.kernel_size, stride=self.stride, padding=self.padding)
        delta_W = outer.mean(dim=0)
        delta_b = torch.einsum("bc,bk->bkc", grad_output.sum(dim=(2, 3)), output_error).mean(dim=0) #Is this right??
        set_or_add_grad(self.weight, delta_W)
        set_or_add_grad(self.bias, delta_b)
                
"""
Vectorized locally connected layer
"""

class Local2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, h_in, w_in, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.h_in = h_in
        self.w_in = w_in
        self.stride = stride
        self.padding = padding
        self.has_bias = bias
        h_out = int(np.floor(((h_in + 2*padding - kernel_size)/stride) + 1))
        w_out = int(np.floor(((w_in + 2*padding - kernel_size)/stride) + 1))
        self.h_out = h_out
        self.w_out = w_out
        k = in_channels*kernel_size**2
        self.weight = nn.Parameter(torch.randn(h_out, w_out, out_channels, in_channels, kernel_size, kernel_size)/np.sqrt(k))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels, h_out, w_out))
        
    def forward(self, input):
        #input = (batch, in_channels, h_in, w_in)
        batch_size = input.shape[0]
        if self.padding > 0:
            padder = nn.ZeroPad2d(self.padding)
            padded_input = padder(input)
        else:
            padded_input = input
        output = torch.zeros(batch_size, self.out_channels, self.h_out, self.w_out, device=input.device)
        for i in range(self.h_out):
            for j in range(self.w_out):
                i1 = i*self.stride
                i2 = i1 + self.kernel_size
                j1 = j*self.stride
                j2 = j1 + self.kernel_size
                input_chunk = padded_input[:, :, i1:i2, j1:j2]
                weight_for_chunk = self.weight[i, j] #, :, :, :]
                output[:, :, i, j] = torch.einsum("oikl,bikl->bo", weight_for_chunk, input_chunk)
        if self.has_bias:
            output = output + self.bias[None, :, :, :]
        return output
    
    
class VecLocal2d(nn.Module):
    def __init__(self, category_dim, in_channels, out_channels, kernel_size, h_in, w_in,
                 stride=1, padding=0,
                 mono=False, first_layer=False, device="cpu"):
            super().__init__()
            self.category_dim = category_dim
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernel_size = kernel_size
            self.h_in = h_in
            self.w_in = w_in
            self.mono = mono
            self.first_layer = first_layer
            self.device = device
            self.stride = stride
            self.padding = padding
            self.lc = Local2d(in_channels, out_channels, kernel_size, h_in, w_in,
                              stride=stride, padding=padding, bias=False).to(device)
            h_out, w_out = self.lc.h_out, self.lc.w_out
            self.h_out = h_out
            self.w_out = w_out
            self.bias = nn.Parameter(torch.zeros(category_dim, out_channels, h_out, w_out, device=device))
            with torch.no_grad():
                init_local(self.lc.weight, first_layer=first_layer, mono=mono)
                if first_layer:
                    self.lc.weight *= np.sqrt(category_dim)

    @property
    def weight(self):
        return self.lc.weight

    def forward(self, input):
        #input = (batch_dim, category_dim, channels, width, height)
        self.input = input.detach()
        batch_size, category_dim = input.shape[:2]
        CWH = input.shape[2:]
        input_reshaped = input.view((batch_size*category_dim,) + CWH)
        output_reshaped = self.lc(input_reshaped)
        output = output_reshaped.view((batch_size, category_dim) + output_reshaped.shape[1:])
        output = output + self.bias[None, :, :, :, :]
        self.mask_shape = (output.shape[0],) + output.shape[2:]
        return output
    
    def post_step_callback(self):
        if self.mono and (not self.first_layer):
            with torch.no_grad():
                #self.conv.weight.clamp_(min=0)
                self.lc.weight.abs_()

    def set_grad(self, activation_mask, output_error):
        #activation_mask = (batch_size, out_channels, h_out, w_out)
        #output_error = (batch_size, category_dim)
        #weight = (h_out, w_out, out_channels, in_channels, kernel_size, kernel_size)
        activation_mask = activation_mask.detach()
        if activation_mask.dtype != torch.float:
            activation_mask = activation_mask.float()
        output_error = output_error.detach()
        input_dp = torch.einsum("bkchw,bk->bchw", self.input, output_error)
        if self.padding > 0:
            padder = nn.ZeroPad2d(self.padding)
            padded_input_dp = padder(input_dp)
        else:
            padded_input_dp = input_dp
        delta_W = torch.zeros_like(self.lc.weight)
        delta_b = torch.zeros_like(self.bias)
        for i in range(self.h_out):
            for j in range(self.w_out):
                i1 = i*self.stride
                i2 = i1 + self.kernel_size
                j1 = j*self.stride
                j2 = j1 + self.kernel_size
                input_chunk = padded_input_dp[:, :, i1:i2, j1:j2] #batch, in_channels, K, K
                post_chunk = activation_mask[:, :, i, j] #batch, out_channels
                delta_W_local = torch.einsum("bikl,bo->oikl", input_chunk, post_chunk) / padded_input_dp.shape[0] #divide by batch size
                delta_b_local = torch.einsum("bo,bk->ko", post_chunk, output_error) / padded_input_dp.shape[0] #divide by batch size
                delta_W[i, j] = delta_W_local
                delta_b[:, :, i, j] = delta_b_local
        set_or_add_grad(self.lc.weight, delta_W)
        set_or_add_grad(self.bias, delta_b)


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
        mask = ((input * self.t[None]).sum(dim=1) >= 0.).float()
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
        mask = ((input * self.t[None]).sum(dim=1) >= 0.).float()
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
        batch_size, category_dim = input.shape[:2]
        CWH = input.shape[2:]
        input_reshaped = input.view((batch_size*category_dim,) + CWH)
        output_reshaped = self.pool(input_reshaped)
        output = output_reshaped.view((batch_size, category_dim) + output_reshaped.shape[1:])
        #output = output * kernel_size #since half the inputs will be zero at initialization
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

def post_step_callback(model):
    for module in model:
        if hasattr(module, "post_step_callback"):
            module.post_step_callback()

"""
Not in use
"""

class cnReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        #input_norm = input.detach() / input.detach().norm(dim=1, keepdim=True)
        #p = input_norm**2
        #ent = (p * torch.log(p)).sum(dim=1)
        #mask = (ent < -1.69).float()
        #print(mask.mean())
        n = input.detach().norm(dim=1) #batch, C, W, H
        n_flat = n.view(n.shape[0], n.shape[1], np.prod(n.shape[2:]))
        #print(n.shape, n_flat.median(dim=2)[0].shape)
        mask = (n >= n_flat.median(dim=2)[0][:, :, None, None]).float()
        self.mask = mask
        #print(mask.shape)
        output = input * mask[:, None]
        return output

    def post_step_callback(self):
        pass

class ReLU(nn.Module):
    def forward(self, input):
        mask = (input.detach().sum(dim=1) >= 0.).float()
        self.mask = mask
        output = input * mask[:, None]
        return output

    def post_step_callback(self):
        pass

class BatchNorm2d(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features)
    def forward(self, input):
        input_reshaped = input.view((input.shape[0]*input.shape[1],) + input.shape[2:])
        output_reshaped = self.bn(input_reshaped)
        output = output_reshaped.view(input.shape)
        return output
    def post_step_callback(self):
        pass

class Dropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    def forward(self, input):
        if self.training:
            mask_shape = (input.shape[0],) + input.shape[2:]
            mask = (torch.rand(mask_shape, device=input.device) > self.p).float()
            output = input * mask[:, None] * (1. / (1. - self.p))
            return output
        else:
            return input
    def post_step_callback(self):
        pass


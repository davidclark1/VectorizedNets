import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def expand_input_conv(input, category_dim):
    #input = (batch, channels, width, height)
    batch_size, in_channels = input.shape[:2]
    expanded_input = torch.zeros((batch_size, category_dim, in_channels*category_dim) + input.shape[2:])
    for i in range(category_dim):
        expanded_input[:, i, i*in_channels:(i+1)*in_channels] = input
    return expanded_input

def expand_input(input, category_dim):
    batch_dim, input_dim = input.shape
    expanded_input = torch.zeros(batch_dim, category_dim, category_dim*input_dim)
    for i in range(category_dim):
        expanded_input[:, i, i*input_dim:(i+1)*input_dim] = input
    return expanded_input

def init_linear(weight, first_layer=False, mono=False):
    if mono:
        return init_linear_mono(weight, first_layer)
    weight *= 0.
    out_features, in_features = weight.shape
    f = 1 if first_layer else 0.5
    weight.normal_(0., 1./np.sqrt(f * in_features))

def init_linear_mono(weight, first_layer):
    if first_layer:
        return init_linear_mono_l0(weight)
    weight *= 0.
    out_features, in_features = weight.shape
    for i in range(max(out_features // 2, 1)):
        W = torch.randn(in_features // 2) / np.sqrt(0.25 * in_features)
        weight[2*i, ::2] = F.relu(W)
        weight[2*i, 1::2] = F.relu(-W)
        if out_features == 1:
            break
        weight[2*i + 1, 1::2] = F.relu(W)
        weight[2*i + 1, ::2] = F.relu(-W)

def init_linear_mono_l0(weight):
    weight *= 0.
    out_features, in_features = weight.shape
    for i in range(out_features // 2):
        W = torch.randn(in_features) / np.sqrt(in_features)
        weight[2*i] = W
        weight[2*i+1] = -W

def init_conv(weight, first_layer=False, mono=False):
    if mono:
        return init_conv_mono(weight, first_layer)
    weight *= 0.
    out_channels, in_channels, kernel_size = weight.shape[:3] #assumes square kernel
    f = 1 if first_layer else 0.5
    weight.normal_(0., 1./np.sqrt(f * in_channels * kernel_size**2))

def init_conv_mono(weight, first_layer):
    if first_layer:
        return init_conv_mono_l0(weight)
    weight *= 0.
    out_channels, in_channels, kernel_size = weight.shape[:3] #assumes square kernel
    filter_shape_2d = weight.shape[2:]
    for i in range(out_channels // 2):
        for j in range(in_channels // 2):
            W = torch.randn(filter_shape_2d) / np.sqrt(0.25 * in_channels * kernel_size**2)
            i1, i2 = i*2, i*2 + 1
            j1, j2 = j*2, j*2 + 1
            weight[i1, j1] = F.relu(W)
            weight[i2, j2] = F.relu(W)
            weight[i1, j2] = F.relu(-W)
            weight[i2, j1] = F.relu(-W)

def init_conv_mono_l0(weight):
    weight *= 0.
    out_channels, in_channels, kernel_size = weight.shape[:3] #assumes square kernel
    filter_shape_3d = weight.shape[1:]
    for i in range(out_channels // 2):
        W = torch.randn(filter_shape_3d) / np.sqrt(in_channels * kernel_size**2)
        weight[2*i] = W
        weight[2*i + 1] = -W

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
        self.weight = nn.Parameter(torch.zeros(out_features, in_features, device=device))
        self.bias = nn.Parameter(torch.zeros(category_dim, out_features, device=device))
        with torch.no_grad():
            init_linear(self.weight, first_layer=first_layer, mono=mono)
            if first_layer:
                self.weight *= np.sqrt(category_dim)

    def forward(self, input):
        self.input = input.detach()
        output = torch.matmul(input, self.weight.T) + self.bias
        self.mask_shape = (output.shape[0],) + output.shape[2:]
        return output

    def post_step_callback(self):
        if self.mono and (not self.first_layer):
            with torch.no_grad():
                self.weight.clamp_(min=0)

    def set_grad(self, activation_mask, output_error):
        #i = batch dim
        #j = category dim
        #n = input feature dim
        #m = output feature dim
        activation_mask = activation_mask.detach()
        output_error = output_error.detach()
        dot_prods = torch.einsum("ijn,ij->in", self.input, output_error)
        delta_W = torch.einsum("im,in->mn", activation_mask, dot_prods) / len(self.input)
        delta_b = torch.einsum("ij,im->jm", output_error, activation_mask) / len(self.input)
        with torch.no_grad():
            if self.weight.grad is None:
                self.weight.grad = delta_W
            else:
                self.weight.grad += delta_W
            if self.bias.grad is None:
                self.bias.grad = delta_b
            else:
                self.bias.grad += delta_b

class tReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.t = None

    def forward(self, input):
        if self.t is None:
            t = torch.randint(0, 2, input.shape[1:], device=input.device).float()*2 - 1
            #t = torch.rand(input.shape[1:], device=input.device)
            #t = (t == t.max(dim=0)[0]).float().to(input.device)
            #t = torch.ones(input.shape[1:], device=input.device).float()
            features = t.shape[1]
            for i in range(features // 2):
                t[:, 2*i + 1] = -t[:, 2*i]
            self.t = t
            print("Instantiated t with shape {}".format(tuple(self.t.shape)))
        mask = ((input.detach() * self.t[None]).sum(dim=1) >= 0.).float()
        self.mask = mask
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

class Flatten(nn.Module):
    def forward(self, input):
        #print("Here")
        #(batch, cat, channels, width, height)
        input_reshaped = input.permute(0, 1, 3, 4, 2).reshape(input.shape[:2] + (np.prod(input.shape[2:]),))
        return input_reshaped
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


def convolutional_outer_product(x1, x2, kernel_size, stride=1):
    #assumes stride, padding, dilation are all defaults!
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

    #x2 = x2.permute(0, 1, 3, 2)
    #w_out_expected = w_in - kernel_size + 1
    #h_out_expected = h_in - kernel_size + 1
    #if (h_out_expected != h_out) or (w_out_expected != w_out):
    #    raise ValueError("invalid kernel size")
    x1_prime = x1.permute(1, 0, 2, 3)
    x2_prime = x2.view(batch_size*out_channels, h_out, w_out).unsqueeze(1)
    out_prime = F.conv2d(input=x1_prime, weight=x2_prime, groups=batch_size, dilation=stride, stride=1)
    out_prime = out_prime[:, :, :kernel_size, :kernel_size]
    out = out_prime.view(in_channels, batch_size, out_channels, kernel_size, kernel_size).permute(1, 2, 0, 3, 4)
    return out



class Conv2d(nn.Module):
    def __init__(self, category_dim, in_channels, out_channels, kernel_size, mono=False, first_layer=False, device="cpu", **conv_params):
            super().__init__()
            self.category_dim = category_dim
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernel_size = kernel_size
            self.mono = mono
            self.first_layer = first_layer
            self.device = device
            self.stride = 1
            for k in conv_params.keys():
                setattr(self, k, conv_params[k]) #will overwrite stride if passed
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, **conv_params).to(device)
            self.bias = nn.Parameter(torch.zeros(category_dim, out_channels, device=device))
            with torch.no_grad():
                init_conv(self.conv.weight, first_layer=first_layer, mono=mono)
                if first_layer:
                    self.conv.weight *= np.sqrt(category_dim)

    @property
    def weight(self):
        return self.conv.weight
    

    def forward(self, input):
        #input = (batch_dim, category_dim, channels, width, height)
        self.input = input.detach()
        batch_size, category_dim = input.shape[:2]
        CWH = input.shape[2:]
        input_reshaped = input.view((batch_size*category_dim,) + CWH)
        output_reshaped = self.conv(input_reshaped)
        output = output_reshaped.view((batch_size, category_dim) + output_reshaped.shape[1:])
        output = output + self.bias[None, :, :, None, None]
        self.mask_shape = (output.shape[0],) + output.shape[2:]
        return output
    
    def post_step_callback(self):
        if self.mono and (not self.first_layer):
            with torch.no_grad():
                self.conv.weight.clamp_(min=0)

    def set_grad(self, activation_mask, output_error, pool_kernel_size=1):
        #activation_mask = (batch_size, out_channels, h_out, w_out)
        #output_error = (batch_size, category_dim)
        activation_mask = activation_mask.detach()
        if activation_mask.dtype != torch.float:
            activation_mask = activation_mask.float()
        output_error = output_error.detach()
        x1 = torch.einsum("bkchw,bk->bchw", self.input, output_error)
        x2 = activation_mask

        #x2 = (torch.rand(x2.shape, device=x2.device ) < x2.mean()).float().to(0)
        outer = convolutional_outer_product(x1, x2, self.kernel_size, stride=self.stride)
        #print(outer)
        delta_W = outer.mean(dim=0) / pool_kernel_size**2
        #print(delta_W.max())
        #delta_W *= 0.
        delta_b = torch.ones(self.bias.shape, device=self.bias.device) * output_error.mean(dim=0)[:, None]
        #f = activation_mask.shape[-1]**2
        #delta_W *= 0.25 * f
        #print("Here")
        with torch.no_grad():
            if self.conv.weight.grad is None:
                self.conv.weight.grad = delta_W
            else:
                self.conv.weight.grad += delta_W
            if self.bias.grad is None:
                self.bias.grad = delta_b
            else:
                self.bias.grad += delta_b

class AvgPool2d(nn.Module):
    def __init__(self, kernel_size, **pool_params):
        super().__init__()
        self.kernel_size = kernel_size
        self.pool = nn.AvgPool2d(kernel_size, **pool_params)

    def forward(self, input):
        batch_size, category_dim = input.shape[:2]
        CWH = input.shape[2:]
        input_reshaped = input.view((batch_size*category_dim,) + CWH)
        output_reshaped = self.pool(input_reshaped)
        output = output_reshaped.view((batch_size, category_dim) + output_reshaped.shape[1:])
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


def set_model_grads(model, output, labels):
    targets = torch.eye(10, device=labels.device)[labels.detach()]
    output_error = F.softmax(output, dim=1) - targets
    for i in range(len(model)):
        layer = model[i]
        if layer.__class__.__name__ in ('Conv2d', 'Linear'):
            #print("Here")
            if (i < len(model) - 1) and (model[i + 1].__class__.__name__ in ('ReLU', 'tReLU')):
                mask = model[i + 1].mask
            else:
                mask = torch.ones(layer.mask_shape, device=output.device)
            layer.set_grad(mask, output_error)
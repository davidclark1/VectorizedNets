import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Linear(nn.Module):
    def __init__(self, category_dim, in_features, out_features, nonneg=False, expanded_input=False, device="cpu"):
        super().__init__()
        self.category_dim = category_dim
        self.in_features = in_features
        self.out_features = out_features
        self.nonneg = nonneg
        self.expanded_input = expanded_input
        self.device = device
        k = 1. / in_features
        k *= 0.2
        if expanded_input:
            k = k * category_dim
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features, device=device))
        with torch.no_grad():
            if nonneg:
                self.weight.uniform_(0, np.sqrt(k))
            else:
                self.weight.uniform_(-np.sqrt(k), np.sqrt(k))
        self.bias = nn.Parameter(torch.zeros(category_dim, out_features, device=device))

    def forward(self, input):
        self.input = input.detach()
        output = torch.matmul(input, self.weight.T) + self.bias
        self.mask_shape = (output.shape[0],) + output.shape[2:]
        return output

    def post_step_callback(self):
        if self.nonneg:
            with torch.no_grad():
                #self.weight.clamp_(min=0)
                self.weight.abs_()
                #self.weight[:] = F.softplus(self.weight[:])

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
            self.t = torch.randint(0, 2, input.shape[1:], device=input.device).float()*2 - 1
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
        #(batch, cat, channels, width, height)
        input_reshaped = input.view(input.shape[:2] + (np.prod(input.shape[2:]),))
        return input_reshaped
    def post_step_callback(self):
        pass

def convolutional_outer_product(x1, x2, kernel_size):
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
    
    w_out_expected = w_in - kernel_size + 1
    h_out_expected = h_in - kernel_size + 1
    if (h_out_expected != h_out) or (w_out_expected != w_out):
        raise ValueError("invalid kernel size")

    x1_prime = x1.permute(1, 0, 2, 3)
    x2_prime = x2.view(batch_size*out_channels, h_out, w_out).unsqueeze(1)
    out_prime = F.conv2d(input=x1_prime, weight=x2_prime, groups=batch_size)
    out = out_prime.view(in_channels, batch_size, out_channels, kernel_size, kernel_size).permute(1, 2, 0, 3, 4)
    return out

class Conv2d(nn.Module):
    def __init__(self, category_dim, in_channels, out_channels, kernel_size, nonneg=False, expanded_input=False, device="cpu", **conv_params):
            super().__init__()
            self.category_dim = category_dim
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernel_size = kernel_size
            self.nonneg = nonneg
            self.expanded_input = expanded_input
            self.device = device
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, **conv_params).to(device)
            k = 1. / (in_channels * np.prod(kernel_size))
            if "groups" in conv_params.keys():
                k = k * groups
            if expanded_input:
                k = k * category_dim
            k *= 0.2
            with torch.no_grad():
                if nonneg:
                    self.conv.weight.uniform_(0, np.sqrt(k))
                else:
                    self.conv.weight.uniform_(-np.sqrt(k), np.sqrt(k))
            self.bias = nn.Parameter(torch.zeros(category_dim, out_channels, device=device))

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
        if self.nonneg:
            with torch.no_grad():
                #self.conv.weight.clamp_(min=0)
                self.conv.weight.abs_()
                #self.conv.weight[:] = F.softplus(self.conv.weight[:])

    def set_grad(self, activation_mask, output_error, pool_kernel_size=2):
        #activation_mask = (batch_size, out_channels, h_out, w_out)
        #output_error = (batch_size, category_dim)
        activation_mask = activation_mask.detach()
        if activation_mask.dtype != torch.float:
            activation_mask = activation_mask.float()
        output_error = output_error.detach()
        x1 = torch.einsum("bkchw,bk->bchw", self.input, output_error)
        x2 = activation_mask
        outer = convolutional_outer_product(x1, x2, self.kernel_size)
        delta_W = outer.mean(dim=0) / pool_kernel_size**2
        delta_b = torch.ones(self.bias.shape, device=self.bias.device) * output_error.mean(dim=0)[:, None]
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


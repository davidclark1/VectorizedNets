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
        output = torch.matmul(input, self.weight.T) + self.bias
        return output

    def post_step_callback(self):
        if self.nonneg:
            with torch.no_grad():
                self.weight.clamp_(min=0)


class tReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.t = None

    def forward(self, input):
        if self.t is None:
            self.t = torch.randint(0, 2, input.shape[1:], device=input.device).float()*2 - 1
            print("Instantiated t with shape {}".format(tuple(self.t.shape)))
        mask = ((input.detach() * self.t[None]).sum(dim=1) >= 0.).float()
        output = input * mask[:, None]
        return output

    def post_step_callback(self):
        pass


class ReLU(nn.Module):
    def forward(self, input):
        mask = (input.detach().sum(dim=1) >= 0.).float()
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
            with torch.no_grad():
                if nonneg:
                    self.conv.weight.uniform_(0, np.sqrt(k))
                else:
                    self.conv.weight.uniform_(-np.sqrt(k), np.sqrt(k))
            self.bias = nn.Parameter(torch.zeros(category_dim, out_channels, device=device))

    def forward(self, input):
        #(batch_dim, category_dim, channels, width, height)
        batch_size, category_dim = input.shape[:2]
        CWH = input.shape[2:]
        input_reshaped = input.view((batch_size*category_dim,) + CWH)
        output_reshaped = self.conv(input_reshaped)
        output = output_reshaped.view((batch_size, category_dim) + output_reshaped.shape[1:])
        output = output + self.bias[None, :, :, None, None]
        return output
    
    def post_step_callback(self):
        if self.nonneg:
            with torch.no_grad():
                self.conv.weight.clamp_(min=0)

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


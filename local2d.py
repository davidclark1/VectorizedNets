import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def lc_forward(input, weight, bias=None, stride=1, padding=0):
    #input = (batch_size, in_channels, h_in, w_in)
    #weight = (out_channels, h_out, w_out, in_channels, kernel_size, kernel_size)
    out_channels, h_out, w_out, in_channels, kernel_size = weight.shape[:-1]
    batch_size = input.shape[0]
    if padding > 0:
        padder = nn.ZeroPad2d(padding)
        padded_input = padder(input)
    else:
        padded_input = input
    output = torch.zeros(batch_size, out_channels, h_out, w_out, device=input.device)
    for i in range(h_out):
        for j in range(w_out):
            i1 = i*stride
            i2 = i1 + kernel_size
            j1 = j*stride
            j2 = j1 + kernel_size
            input_chunk = padded_input[:, :, i1:i2, j1:j2]
            weight_for_chunk = weight[:, i, j]
            output[:, :, i, j] += torch.einsum("oikl,bikl->bo", weight_for_chunk, input_chunk)
    if bias is not None:
        output = output + bias
    return output
    
def lc_backward(grad_output, weight, input_shape, stride=1, padding=0):
    #weight = (out_channels, h_out, w_out, in_channels, K, K)
    in_channels, h_in, w_in = input_shape
    batch_size, out_channels = grad_output.shape[:2]
    out_channels, h_out, w_out, in_channels, kernel_size = weight.shape[:5]
    grad_input = torch.zeros((batch_size, in_channels, h_in+2*padding, w_in+2*padding), device=grad_output.device)
    for k in range(kernel_size):
        for l in range(kernel_size):
            relevant_weight = weight[:, :, :, :, k, l]
            result = torch.einsum("omnc,bomn->bcmn", relevant_weight, grad_output)
            s1, s2 = result.shape[-2:]
            #grad_input[:, :, stride*k:stride*k+s1, stride*l:stride*l+s2] += result
            grad_input[:, :, k:k+s1*stride:stride, l:l+s2*stride:stride] += result
    if padding > 0:
        #s1, s2 = grad_input.shape[-2:]
        #grad_input = grad_input[:, :, padding:s1-padding, padding:s2-padding]
        grad_input = grad_input[:, :, padding:-padding, padding:-padding]
    return grad_input

def lc_compute_grads(input, grad_output, kernel_size, bias=False, stride=1, padding=0, output_error=None):
        if padding > 0:
            padder = nn.ZeroPad2d(padding)
            padded_input = padder(input)
        else:
            padded_input = input
        in_channels, h_in, w_in = input.shape[1:]
        out_channels, h_out, w_out = grad_output.shape[1:]
        delta_W = torch.zeros(out_channels, h_out, w_out, in_channels, kernel_size, kernel_size, device=input.device)
        if bias:
            if output_error is not None:
                delta_b =  torch.zeros(output_error.shape[1], out_channels, h_out, w_out, device=input.device)
            else:
                delta_b =  torch.zeros(out_channels, h_out, w_out, device=input.device)
        else:
            delta_b = None
        for i in range(h_out):
            for j in range(w_out):
                i1 = i*stride
                i2 = i1 + kernel_size
                j1 = j*stride
                j2 = j1 + kernel_size
                input_chunk = padded_input[:, :, i1:i2, j1:j2] #batch, in_channels, K, K
                post_chunk = grad_output[:, :, i, j] #batch, out_channels
                delta_W_local = torch.einsum("bikl,bo->oikl", input_chunk, post_chunk) #/ padded_input.shape[0] #divide by batch size
                delta_W[:, i, j] = delta_W_local
                if bias:
                    if output_error is not None:
                        delta_b_local = torch.einsum("bo,bc->co", post_chunk, output_error)
                    else:
                        delta_b_local = post_chunk.sum(dim=0)
                    delta_b[..., i, j] = delta_b_local
        return delta_W, delta_b #return both grads

class Local2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0):
        ctx.save_for_backward(input, weight)
        ctx.has_bias = bias is not None
        ctx.stride = stride
        ctx.padding = padding
        output = lc_forward(input, weight, bias=bias, stride=stride, padding=padding)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        has_bias, stride, padding = ctx.has_bias, ctx.stride, ctx.padding
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = lc_backward(grad_output, weight, input.shape, stride=stride, padding=padding)
        if ctx.needs_input_grad[1]:
            grad_weight, grad_bias = lc_compute_grads(input, grad_output, weight.shape[-1],
                bias=has_bias, stride=stride, padding=padding)
        return grad_input, grad_weight, grad_bias, None, None

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
        self.weight = nn.Parameter(torch.randn(out_channels, h_out, w_out, in_channels, kernel_size, kernel_size)/np.sqrt(k))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels, h_out, w_out))
        else:
            self.register_parameter('bias', None)
        
    def forward(self, input):
        return Local2dFunction.apply(input, self.weight, self.bias, self.stride, self.padding)

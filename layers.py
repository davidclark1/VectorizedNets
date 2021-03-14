import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F

class VectorizedLayer(nn.Module):
    def __init__(self, in_features, out_features, category_dim, nonneg=False, nonlin=True, expanded_input=False):
        super(VectorizedLayer, self).__init__()
        k = 1. / np.sqrt(in_features)
        if expanded_input:
            k = 1. / np.sqrt(in_features / category_dim)
        k = k * 0.25
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        with torch.no_grad():
            if nonneg: self.weight.uniform_(0, k)
            else: self.weight.uniform_(-k, k)
        self.bias = nn.Parameter(torch.zeros(category_dim, out_features))
        self.nonneg = nonneg
        self.nonlin = nonlin
        self.mask_weights = torch.randint(0, 2, (category_dim, out_features))*2 - 1
        #self.mask_weights = torch.ones(category_dim, out_features)
        #for i in range(out_features):
        #    if np.random.rand() < 0.5:
        #        self.mask_weights[:, i] = -1
        
    def forward(self, input):
        self.input = input.detach()
        h = torch.matmul(input, self.weight.T) + self.bias
        if self.nonlin:
            mask = ((h.detach() * self.mask_weights).sum(dim=1) >= 0.).float()
            self.mask = mask
            h = h * mask[:, None, :]
        else:
            self.mask = torch.ones(h.shape[0], h.shape[2])
        return h
    
    def update(self, error, eta):
        #i = batch dim
        #j = category dim
        #n = input feature dim
        #m = output feature dim
        dot_prods = torch.einsum("ijn,ij->in", self.input, error.detach())
        delta_weight = torch.einsum("im,in->mn", self.mask, dot_prods) / len(self.input)
        delta_bias = torch.einsum("ij,im->jm", error, self.mask) / len(self.input)
        with torch.no_grad():
            self.weight -= eta*delta_weight
            self.bias -= eta*delta_bias
        self.post_step_callback()
        
    def set_grad(self, error):
        #i = batch dim
        #j = category dim
        #n = input feature dim
        #m = output feature dim
        dot_prods = torch.einsum("ijn,ij->in", self.input, error.detach())
        delta_weight = torch.einsum("im,in->mn", self.mask, dot_prods) / len(self.input)
        delta_bias = torch.einsum("ij,im->jm", error, self.mask) / len(self.input)
        
        if self.weight.grad is None:
            self.weight.grad = delta_weight.detach()
        else:
            self.weight.grad += delta_weight.detach()
        if self.bias.grad is None:
            self.bias.grad = delta_bias.detach()
        else:
            self.bias.grad += delta_bias.detach()
        
    def post_step_callback(self):
        if self.nonneg:
            with torch.no_grad():
                self.weight.clamp_(min=0)
                
def expand_input(input, category_dim):
    batch_dim, input_dim = input.shape
    expanded_input = torch.zeros(batch_dim, category_dim, category_dim*input_dim)
    for i in range(category_dim):
        expanded_input[:, i, i*input_dim:(i+1)*input_dim] = input
    return expanded_input

def permute_input(input, category_dim):
    batch_dim, input_dim = input.shape
    permuted_input = torch.zeros(batch_dim, category_dim, input_dim)
    for i in range(category_dim):
        permuted_input[:, i, :] = input[:, PERMUTATIONS[i]]
    return permuted_input

def eval_test_accuracy(model, input_data_fn=permute_input):
    num_correct = 0
    for batch_idx, (data, labels) in enumerate(test_loader):
        input = input_data_fn(data, 10)
        #with torch.no_grad():
        out = model.forward(input)[..., 0]
        num_correct += (out.argmax(dim=1) == labels).int().sum().item()
    acc = num_correct / 10000.
    return acc


import local2d
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DFAWrapper(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.sequential = nn.Sequential(*layers)
        
    def forward(self, input, learning_rule):
        x = input
        if learning_rule == "bp":
            for layer in self.sequential:
                if type(layer).__name__ not in ('DFA', 'DFALayer'):
                    x = layer(x)
        elif learning_rule == "df":
            x = self.sequential(x)
        return x
    
def make_dfas(n, mono):
    dfas = [mono_dfa.DFALayer() for _ in range(n)]
    dfa = mono_dfa.DFA(dfas, normalization=False, mono=mono)
    return dfas + [dfa]

def rectify_grads(model):
    for i, layer in enumerate(model.sequential):
        if i > 0 and type(layer).__name__ in ('Linear', 'Conv2d', 'Local2d'):
            layer.weight.clamp_(min=0.)

def post_step_callback(model):
    last_dfa = model.sequential[-1]
    if last_dfa.mono:
        rectify_grads(model)

def zero_biases(model):
    for layer in model.sequential:
        if type(layer).__name__ in ('Linear', 'Conv2d', 'Local2d'):
            layer.bias[:] = 0.


def init_model(model, mono):
    with torch.no_grad():
        for i, layer in enumerate(model.sequential):
            first_layer = i == 0
            if type(layer).__name__ =='Linear':
                vnn.init_linear(layer.weight, mono=mono, first_layer=first_layer)
                layer.bias[:] = 0.
            elif type(layer).__name__ =='Conv2d':
                vnn.init_conv(layer.weight, mono=mono, first_layer=first_layer)
                layer.bias[:] = 0.
            elif type(layer).__name__ =='Local2d':
                vnn.init_lc(layer.weight, mono=mono, first_layer=first_layer)
                layer.bias[:] = 0.


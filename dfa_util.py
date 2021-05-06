import local2d
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import init_methods

class DFABackend(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dfa_context):
        ctx.dfa_context = dfa_context  # Access to global informations in the backward
        return input

    @staticmethod
    def backward(ctx, grad_output):
        dfa_context = ctx.dfa_context

        if not dfa_context.no_training:
            random_projection = torch.mm(grad_output.reshape(grad_output.shape[0], -1).to(dfa_context.rp_device),
                                         dfa_context.feedback_matrix)  # Global random projection
            if dfa_context.normalization:
                random_projection /= np.sqrt(np.prod(random_projection.shape[1:]))

            dfa_context.random_projection = random_projection

        return grad_output, None  # Gradients for output and dfa_context (None)


class DFA(nn.Module):
    def __init__(self, dfa_layers, normalization=True, rp_device=None, no_training=False,
        mono=False):
        super(DFA, self).__init__()
        self.mono = mono

        self.dfa_layers = dfa_layers

        for dfa_layer in self.dfa_layers:
            dfa_layer.hook_function = self._hook

        self.normalization = normalization
        self.rp_device = rp_device
        self.no_training = no_training

        self.dfa = DFABackend.apply  # Custom DFA autograd function that actually handles the backward

        # Random feedback matrix and its dimensions
        self.feedback_matrix = None
        self.max_feedback_size = 0
        self.output_size = 0

        self.initialized = False

        self.random_projection = None

    def forward(self, input):
        if not(self.initialized or self.no_training):
            if self.rp_device is None:
                self.rp_device = input.device

            self.output_size = np.prod(input.shape[1:])

            for layer in self.dfa_layers:
                feedback_size = np.prod(layer.feedback_size)
                if feedback_size > self.max_feedback_size:
                    self.max_feedback_size = feedback_size

            if self.mono:
                self.feedback_matrix = nn.Parameter(torch.rand(self.output_size, self.max_feedback_size, device=self.rp_device), requires_grad=False)
            else:
                self.feedback_matrix = nn.Parameter(2*torch.rand(self.output_size, self.max_feedback_size, device=self.rp_device)-1, requires_grad=False)

            self.initialized = True

        return self.dfa(input, self)

    def _hook(self, grad):
        if self.random_projection is not None:
            return self.random_projection[:, :np.prod(grad.shape[1:])].view(*grad.shape).to(grad.device)
        else:
            grad[:] = 0
            return grad

class DFALayer(nn.Module):
    def __init__(self, name=None, passthrough=False):
        super(DFALayer, self).__init__()

        self.name = name
        self.passthrough = passthrough

        self.hook_function = None

        self.feedback_size = None
        self.initialized = False

    def forward(self, input):
        #input = input + 0.
        if not self.initialized:
            self.feedback_size = input.shape[1:]
            self.initialized = True
        if not self.passthrough and input.requires_grad:
            input.register_hook(self.hook_function)
        return input

class DFAWrapper(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.sequential = nn.Sequential(*layers)
        
    def forward(self, input, learning_rule=None):
        if learning_rule is None:
            learning_rule = "bp"
        x = input
        if learning_rule == "bp":
            for layer in self.sequential:
                if type(layer).__name__ not in ('DFA', 'DFALayer'):
                    x = layer(x)
        elif learning_rule == "df":
            x = self.sequential(x)
        return x
    
def make_dfas(n, mono):
    dfas = [DFALayer() for _ in range(n)]
    dfa = DFA(dfas, normalization=False, mono=mono)
    return dfas + [dfa]

def rectify_grads(model):
    with torch.no_grad():
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
            first_layer = (i == 0)
            if type(layer).__name__ =='Linear':
                init_methods.init_linear(layer.weight, mono=mono, first_layer=first_layer)
                layer.bias[:] = 0.
            elif type(layer).__name__ =='Conv2d':
                init_methods.init_conv(layer.weight, mono=mono, first_layer=first_layer)
                layer.bias[:] = 0.
            elif type(layer).__name__ =='Local2d':
                init_methods.init_local(layer.weight, mono=mono, first_layer=first_layer)
                layer.bias[:] = 0.


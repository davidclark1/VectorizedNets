import dfa
import local2d
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#FC model

def make_mnist_nonvec_fc(mono):
    dfa1, dfa2, dfa = dfa.make_dfas(2, mono)
    model = dfa.DFAWrapper(
        nn.Linear(28*28, 1024),
        nn.ReLU(),
        dfa1,
        nn.Linear(1024, 512),
        nn.ReLU(),
        dfa2,
        nn.Linear(512, 10),
        dfa)
    dfa.init_model(model, mono)
    return model

def make_cifar_nonvec_fc(mono):
    dfa1, dfa2, dfa3, dfa = dfa.make_dfas(3, mono)
    model = dfa.DFAWrapper(
        nn.Linear(32*32*3, 1024),
        nn.ReLU(),
        dfa1,
        nn.Linear(1024, 512),
        nn.ReLU(),
        dfa2,
        nn.Linear(512, 512),
        nn.ReLU(),
        dfa3,
        nn.Linear(512, 10),
        dfa)
    dfa.init_model(model, mono)
    return model

#Conv model

def make_mnist_nonvec_conv(mono):
    dfa1, dfa2, dfa3, dfa = dfa.make_dfas(3, mono)
    model = dfa.DFAWrapper(
        nn.Conv2d(1, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        dfa1,
        nn.AvgPool2d(2),
        nn.Conv2d(64, 32, 3, stride=1, padding=1),
        nn.ReLU(),
        dfa2,
        nn.AvgPool2d(2), #7 by 7
        nn.Flatten(),
        nn.Linear(1568, 1024),
        nn.ReLU(),
        dfa3,
        nn.Linear(1024, 10),
        dfa)
    dfa.init_model(model, mono)
    return model

def make_cifar_nonvec_conv(mono):
    dfa1, dfa2, dfa3, dfa4, dfa = dfa.make_dfas(4, mono)
    model = dfa.DFAWrapper(
        nn.Conv2d(3, 128, 5, stride=1, padding=2),
        nn.ReLU(),
        dfa1,
        nn.AvgPool2d(2),
        nn.Conv2d(128, 64, 5, stride=1, padding=2),
        nn.ReLU(),
        dfa2,
        nn.AvgPool2d(2),
        nn.Conv2d(64, 64, 2, stride=2, padding=0),
        nn.ReLU(),
        dfa3,
        nn.Flatten(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        dfa4,
        nn.Linear(1024, 10),
        dfa)
    dfa.init_model(model, mono)
    return model

#LC model

def make_mnist_nonvec_lc(mono):
    dfa1, dfa2, dfa3, dfa = dfa.make_dfas(3, mono)
    model = dfa.DFAWrapper(
        local2d.Local2d(1, 32, 3, h_in=28, w_in=28, stride=1, padding=1),
        nn.ReLU(),
        dfa1,
        nn.AvgPool2d(2),
        local2d.Local2d(32, 32, 3, h_in=14, w_in=14, stride=1, padding=1),
        nn.ReLU(),
        dfa2,
        nn.AvgPool2d(2), #7 by 7
        nn.Flatten(),
        nn.Linear(1568, 1024),
        nn.ReLU(),
        dfa3,
        nn.Linear(1024, 10),
        dfa)
    dfa.init_model(model, mono)
    return model

def make_cifar_nonvec_lc(mono):
    dfa1, dfa2, dfa3, dfa4, dfa = dfa.make_dfas(4, mono)
    model = dfa.DFAWrapper(
        local2d.Local2d(3, 64, 5, h_in=32, w_in=32, stride=1, padding=2),
        nn.ReLU(),
        dfa1,
        nn.AvgPool2d(2),
        local2d.Local2d(64, 32, 5, h_in=16, w_in=16, stride=1, padding=2),
        nn.ReLU(),
        dfa2,
        nn.AvgPool2d(2),
        local2d.Local2d(32, 32, 2, h_in=8, w_in=8, stride=2, padding=0),
        nn.ReLU(),
        dfa3,
        nn.Flatten(),
        nn.Linear(512, 512),
        nn.ReLU(),
        dfa4,
        nn.Linear(512, 10),
        dfa)
    dfa.init_model(model, mono)
    return model





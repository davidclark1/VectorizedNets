import sys
sys.path.append("..")
import vnn
import local2d
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


#FC model

def make_mnist_vec_fc(mono=False):
    model = nn.Sequential(
        vnn.Linear(10, 28*28*10, 1024, first_layer=True, mono=mono),
        vnn.tReLU(10, 1024),
        vnn.Linear(10, 1024, 512, mono=mono),
        vnn.tReLU(10, 512),
        vnn.Linear(10, 512, 1, mono=mono))
    return model

def make_cifar_vec_fc(mono=False):
    model = nn.Sequential(
        vnn.Linear(10, 32*32*3*10, 1024, first_layer=True, mono=mono),
        vnn.tReLU(10, 1024),
        vnn.Linear(10, 1024, 512, mono=mono),
        vnn.tReLU(10, 512),
        vnn.Linear(10, 512, 512, mono=mono),
        vnn.tReLU(10, 512),
        vnn.Linear(10, 512, 1, mono=mono))
    return model

#Conv model

def make_mnist_vec_conv(mono=False):
    model = nn.Sequential(
        vnn.Conv2d(10, 10, 64, 3, stride=1, padding=1, first_layer=True, mono=mono),
        vnn.ctReLU(10, 64, 28, 28),
        vnn.AvgPool2d(2),
        vnn.Conv2d(10, 64, 32, 3, stride=1, padding=1, mono=mono),
        vnn.ctReLU(10, 32, 14, 14),
        vnn.AvgPool2d(2), #7 by 7
        vnn.Flatten(),
        vnn.Linear(10, 1568, 1024, mono=mono),
        vnn.tReLU(10, 1024),
        vnn.Linear(10, 1024, 1, mono=mono))
    return model

def make_cifar_vec_conv(mono=False):
    model = nn.Sequential(
        vnn.Conv2d(10, 30, 128, 5, stride=1, padding=2, first_layer=True, mono=mono),
        vnn.ctReLU(10, 128, 32, 32),
        vnn.AvgPool2d(2),
        vnn.Conv2d(10, 128, 64, 5, stride=1, padding=2, mono=mono),
        vnn.ctReLU(10, 64, 16, 16),
        vnn.AvgPool2d(2),
        vnn.Conv2d(10, 64, 64, 2, stride=2, padding=0, mono=mono),
        vnn.ctReLU(10, 64, 4, 4),
        vnn.Flatten(),
        vnn.Linear(10, 1024, 1024, mono=mono),
        vnn.tReLU(10, 1024),
        vnn.Linear(10, 1024, 1, mono=mono))
    return model

#LC model

def make_mnist_vec_lc(mono=False):
    model = nn.Sequential(
        vnn.Local2d(10, 10, 32, 3, h_in=28, w_in=28, stride=1, padding=1, first_layer=True, mono=mono),
        vnn.ctReLU(10, 32, 28, 28),
        vnn.AvgPool2d(2),
        vnn.Local2d(10, 32, 32, 3, h_in=14, w_in=14, stride=1, padding=1, mono=mono),
        vnn.ctReLU(10, 32, 14, 14),
        vnn.AvgPool2d(2), #7 by 7
        vnn.Flatten(),
        vnn.Linear(10, 1568, 1024, mono=mono),
        vnn.tReLU(10, 1024),
        vnn.Linear(10, 1024, 1, mono=mono))
    return model

def make_cifar_vec_lc(mono=False):
    model = nn.Sequential(
        vnn.Local2d(10, 30, 64, 5, h_in=32, w_in=32, stride=1, padding=2, first_layer=True, mono=mono),
        vnn.ctReLU(10, 64, 32, 32),
        vnn.AvgPool2d(2),
        vnn.Local2d(10, 64, 32, 5, h_in=16, w_in=16, stride=1, padding=2, mono=mono),
        vnn.ctReLU(10, 32, 16, 16),
        vnn.AvgPool2d(2),
        vnn.Local2d(10, 32, 32, 2, h_in=8, w_in=8, stride=2, padding=0, mono=mono),
        vnn.ctReLU(10, 32, 4, 4),
        vnn.Flatten(),
        vnn.Linear(10, 512, 256, mono=mono),
        vnn.tReLU(10, 256),
        vnn.Linear(10, 256, 1, mono=mono))
    return model




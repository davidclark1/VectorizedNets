import sys
sys.path.append("..")
import vnn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.testing import assert_allclose
import pytest 

@pytest.mark.parametrize("category_dim", [1, 10, 100])
def test_conv(category_dim):
    x = torch.randn(16, category_dim, 3, 32, 32)
    conv = vnn.Conv2d(category_dim=category_dim, in_channels=3, out_channels=12, kernel_size=5, stride=1, padding=2)
    out = conv(x).detach().numpy()
    target_out = torch.zeros(16, category_dim, 12, 32, 32)
    for i in range(category_dim):
        target_out[:, i, :, :, :] = F.conv2d(x[:, i, :, :, :], weight=conv.weight, bias=conv.bias[i], padding=2)
    target_out = target_out.detach().numpy()
    assert_allclose(out, target_out)

@pytest.mark.parametrize("category_dim", [1, 10, 100])
def test_linear(category_dim):
    x = torch.randn(16, category_dim, 200)
    fc = vnn.Linear(category_dim=category_dim, in_features=200, out_features=50)
    out = fc(x).detach().numpy()
    target_out = torch.zeros(16, category_dim, 50)
    for i in range(category_dim):
        target_out[:, i, :] = F.linear(x[:, i, :], weight=fc.weight, bias=fc.bias[i])
    target_out = target_out.detach().numpy()
    assert_allclose(out, target_out, atol=1e-4)
    assert_allclose(out, target_out, rtol=1e-2)

@pytest.mark.parametrize("in_channels, out_channels", [(1, 5), (5, 1), (3, 5), (7, 12)])
def test_nonvectorized_local(in_channels, out_channels):
    x = torch.randn(16, in_channels, 32, 32)
    lc = vnn.Local2d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, h_in=32, w_in=32, stride=1, padding=2)
    out = lc(x).detach().numpy()
    assert(out.shape == (16, out_channels, 32, 32))

@pytest.mark.parametrize("in_channels, out_channels", [(1, 5), (5, 1), (3, 5), (7, 12)])
def test_nonvectorized_local_strided(in_channels, out_channels):
    x = torch.randn(16, in_channels, 32, 32)
    lc = vnn.Local2d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, h_in=32, w_in=32, stride=2, padding=2)
    out = lc(x).detach().numpy()
    assert(out.shape == (16, out_channels, 16, 16))

@pytest.mark.parametrize("in_channels, out_channels, bias", [(1, 5, True), (5, 1, False), (3, 5, True), (7, 12, False)])
def test_nonvectorized_local_bias(in_channels, out_channels, bias):
    x = torch.randn(16, in_channels, 32, 32)
    lc = vnn.Local2d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, h_in=32, w_in=32, stride=1, padding=2, bias=bias)
    out = lc(x).detach().numpy()
    assert(out.shape == (16, out_channels, 32, 32))


@pytest.mark.parametrize("category_dim", [2, 10, 13])
def test_local_shape(category_dim):
    x = torch.randn(16, category_dim, 5, 32, 32)
    lc = vnn.VecLocal2d(category_dim=category_dim, in_channels=5, out_channels=3,
    					kernel_size=5, h_in=32, w_in=32, stride=1, padding=2)
    out = lc(x).detach().numpy()
    assert(out.shape == (16, category_dim, 3, 32, 32))


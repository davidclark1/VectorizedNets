import sys
sys.path.append("..")
import vnn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.testing import assert_allclose
import pytest 


@pytest.mark.parametrize("in_features, out_features", [(2, 8), (4, 2), (50, 6)])
def test_init_linear_1(in_features, out_features):
    W = torch.zeros(out_features, in_features)
    vnn.init_linear(W, first_layer=True, mono=True)
    W = W.numpy()
    assert_allclose(W[::2], -W[1::2])

@pytest.mark.parametrize("in_features, out_features", [(111, 100), (401, 200), (567, 200)])
def test_init_linear_2(in_features, out_features):
    W = torch.zeros(out_features, in_features)
    vnn.init_linear(W, first_layer=True, mono=False)
    W = W.numpy()
    frac_positive = np.mean(W >= 0.)
    assert(np.abs(frac_positive - 0.5) < 0.15)

@pytest.mark.parametrize("in_features, out_features", [(2, 8), (4, 2), (50, 100)])
def test_init_linear_3(in_features, out_features):
    W = torch.zeros(out_features, in_features)
    vnn.init_linear(W, first_layer=False, mono=True)
    W = W.numpy()
    assert(np.all(W >= 0.))
    assert_allclose(W[::2, ::2], W[1::2, 1::2])
    assert_allclose(W[1::2, ::2], W[::2, 1::2])

@pytest.mark.parametrize("in_features, out_features", [(101, 101), (401, 203), (507, 200)])
def test_init_linear_4(in_features, out_features):
    W = torch.zeros(out_features, in_features)
    vnn.init_linear(W, first_layer=False, mono=False)
    W = W.numpy()
    frac_positive = np.mean(W >= 0.)
    assert(np.abs(frac_positive - 0.5) < 0.15)

@pytest.mark.parametrize("in_features, out_features", [(400, 1), (420, 1), (500, 1)])
def test_init_linear_5(in_features, out_features):
    print(out_features)
    W = torch.zeros(out_features, in_features)
    vnn.init_linear(W, first_layer=False, mono=True)
    W = W.numpy()
    assert(np.all(W >= 0.))

@pytest.mark.parametrize("in_channels, out_channels", [(2, 8), (4, 2), (50, 6)])
def test_init_conv_1(in_channels, out_channels):
    W = torch.zeros(out_channels, in_channels, 5, 5)
    vnn.init_conv(W, first_layer=True, mono=True)
    W = W.numpy()
    assert_allclose(W[::2], -W[1::2])

@pytest.mark.parametrize("in_channels, out_channels", [(20, 80), (42, 22), (52, 36)])
def test_init_conv_2(in_channels, out_channels):
    W = torch.zeros(out_channels, in_channels, 5, 5)
    vnn.init_conv(W, first_layer=True, mono=False)
    W = W.numpy()
    frac_positive = np.mean(W >= 0.)
    assert(np.abs(frac_positive - 0.5) < 0.15)

@pytest.mark.parametrize("in_channels, out_channels", [(2, 8), (4, 2), (50, 6)])
def test_init_conv_3(in_channels, out_channels):
    W = torch.zeros(out_channels, in_channels, 5, 5)
    vnn.init_conv(W, first_layer=False, mono=True)
    W = W.numpy()
    assert(np.all(W >= 0.))
    assert_allclose(W[::2, ::2], W[1::2, 1::2])
    assert_allclose(W[1::2, ::2], W[::2, 1::2])

@pytest.mark.parametrize("in_channels, out_channels", [(211, 83), (42, 22), (51, 61)])
def test_init_conv_4(in_channels, out_channels):
    W = torch.zeros(out_channels, in_channels, 5, 5)
    vnn.init_conv(W, first_layer=False, mono=False)
    W = W.numpy()
    frac_positive = np.mean(W >= 0.)
    assert(np.abs(frac_positive - 0.5) < 0.15)

@pytest.mark.parametrize("in_channels, out_channels", [(2, 8), (4, 2), (50, 6)])
def test_init_lc_1(in_channels, out_channels):
    W = torch.zeros(out_channels, 21, 21, in_channels, 5, 5)
    vnn.init_local(W, first_layer=True, mono=True)
    W = W.numpy()
    assert_allclose(W[::2], -W[1::2])

@pytest.mark.parametrize("in_channels, out_channels", [(20, 80), (42, 22), (52, 36)])
def test_init_lc_2(in_channels, out_channels):
    W = torch.zeros(out_channels, 21, 21, in_channels, 5, 5)
    vnn.init_local(W, first_layer=True, mono=False)
    W = W.numpy()
    frac_positive = np.mean(W >= 0.)
    assert(np.abs(frac_positive - 0.5) < 0.15)

@pytest.mark.parametrize("in_channels, out_channels", [(2, 8), (4, 2), (50, 6)])
def test_init_lc_3(in_channels, out_channels):
    W = torch.zeros(out_channels, 21, 21, in_channels, 5, 5)
    vnn.init_local(W, first_layer=False, mono=True)
    W = W.numpy()
    assert(np.all(W >= 0.))
    assert_allclose(W[::2, :, :, ::2], W[1::2, :, :, 1::2])
    assert_allclose(W[1::2, :, :, ::2], W[::2, :, :, 1::2])

@pytest.mark.parametrize("in_channels, out_channels", [(211, 83), (42, 22), (51, 61)])
def test_init_lc_4(in_channels, out_channels):
    W = torch.zeros(out_channels, 21, 21, in_channels, 5, 5)
    vnn.init_local(W, first_layer=False, mono=False)
    W = W.numpy()
    frac_positive = np.mean(W >= 0.)
    assert(np.abs(frac_positive - 0.5) < 0.15)


@pytest.mark.parametrize("channels", [6, 4, 2])
def test_ctrelu_1(channels):
    x = torch.randn(128, 10, channels, 28, 28)
    nonlin = vnn.ctReLU(10, channels, 28, 28, share_t=True)
    y1 = nonlin(x)
    y2 = nonlin(x)
    assert_allclose(y1, y2)
    assert(nonlin.t.shape == (10, channels, 28, 28))
    assert(torch.all(nonlin.t[:, 0, 5, 5] == nonlin.t[:, 0, 20, 20]))

@pytest.mark.parametrize("channels", [6, 4, 2])
def test_ctrelu_2(channels):
    x = torch.randn(128, 20, channels, 28, 28)
    nonlin = vnn.ctReLU(20, channels, 28, 28, share_t=False)
    y1 = nonlin(x)
    y2 = nonlin(x)
    assert_allclose(y1, y2)
    assert(nonlin.t.shape == (20, channels, 28, 28))
    assert(torch.any(nonlin.t[:, 0, 5, 5] != nonlin.t[:, 0, 20, 20])) #should fail ~1/10^6 times


@pytest.mark.parametrize("features", [6, 4, 2])
def test_trelu(features):
    x = torch.randn(128, 10, features)
    nonlin = vnn.tReLU(10, features)
    y1 = nonlin(x)
    y2 = nonlin(x)
    assert_allclose(y1, y2)
    assert(nonlin.t.shape == (10, features))












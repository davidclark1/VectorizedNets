import sys
sys.path.append("..")
import vnn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.testing import assert_allclose
import pytest 

def test_conv_outer_prod_shape():
    x1 = torch.randn(128, 5, 32, 32)
    W = torch.randn(3, 5, 4, 4)
    x2 = F.conv2d(x1, weight=W)
    out = vnn.convolutional_outer_product(x1, x2, kernel_size=4)
    assert(out.shape == (128, 3, 5, 4, 4))

@pytest.mark.parametrize("batch_size, kernel_size", [(128, 3), (128, 5), (256, 3), (256, 5)])
def test_conv_outer_prod(batch_size, kernel_size):
    B, K = batch_size, kernel_size
    x1 = torch.randn(B, 5, 36, 36).double()
    W = torch.randn(3, 5, K, K).double()
    x2 = F.conv2d(x1, weight=W).double()
    out = vnn.convolutional_outer_product(x1, x2, kernel_size=K)
    target_out = torch.zeros(B, 3, 5, K, K).double()
    for i in range(3):
        for j in range(5):
            target_out[:, i, j, :, :] = F.conv2d(x1[:, j:j+1].permute(1, 0, 2, 3), x2[:, i:i+1], groups=B)[0]
    assert_allclose(out, target_out)

@pytest.mark.parametrize("padding", [0, 3, 5, 10])
def test_conv_outer_prod_padding(padding):
    B, K = 32, 5
    x1 = torch.randn(B, 5, 36, 36).double()
    W = torch.randn(3, 5, K, K).double()
    x2 = F.conv2d(x1, weight=W, padding=padding).double()
    out = vnn.convolutional_outer_product(x1, x2, kernel_size=K, padding=padding)
    target_out = torch.zeros(B, 3, 5, K, K).double()
    for i in range(3):
        for j in range(5):
            target_out[:, i, j, :, :] = F.conv2d(x1[:, j:j+1].permute(1, 0, 2, 3), x2[:, i:i+1],
                groups=B, padding=padding)[0]
    assert_allclose(out, target_out)

@pytest.mark.parametrize("stride", [1, 2, 3])
def test_conv_outer_prod_stride(stride):
    B, K = 32, 5
    x1 = torch.randn(B, 5, 36, 36).double()
    W = torch.randn(3, 5, K, K).double()
    x2 = F.conv2d(x1, weight=W, stride=stride).double()
    out = vnn.convolutional_outer_product(x1, x2, kernel_size=K, stride=stride)
    target_out = torch.zeros(B, 3, 5, K, K).double()
    for i in range(3):
        for j in range(5):
            target_out[:, i, j, :, :] = F.conv2d(x1[:, j:j+1].permute(1, 0, 2, 3), x2[:, i:i+1],
                groups=B, dilation=stride)[0, :, :K, :K]
    assert_allclose(out, target_out)

def test_conv_outer_prod_grad():
    B, K = 100, 5
    x1 = torch.randn(B, 5, 36, 36).double()
    W = torch.randn(3, 5, K, K).double()
    W.requires_grad = True
    x2 = F.conv2d(x1, weight=W)
    avg = x2.sum() / B
    avg.backward()
    target_grad = W.grad
    x2 = x2.detach()
    pred_grad = vnn.convolutional_outer_product(x1, torch.ones_like(x2), kernel_size=K).mean(dim=0)
    assert_allclose(target_grad, pred_grad)

@pytest.mark.parametrize("stride", [1, 2, 3, 5, 8])
def test_conv_outer_prod_grad_stride(stride):
    B, K = 100, 5
    x1 = torch.randn(B, 5, 36, 36).double()
    W = torch.randn(3, 5, K, K).double()
    W.requires_grad = True
    x2 = F.conv2d(x1, weight=W, stride=stride)
    avg = x2.sum() / B
    avg.backward()
    target_grad = W.grad
    x2 = x2.detach()
    pred_grad = vnn.convolutional_outer_product(x1, torch.ones_like(x2), kernel_size=K, stride=stride).mean(dim=0)
    assert_allclose(target_grad, pred_grad)

@pytest.mark.parametrize("stride", [1, 2, 3, 5, 8])
def test_conv_outer_prod_grad_relu_stride(stride):
    B, K = 100, 5
    x1 = torch.randn(B, 5, 36, 36).double()
    W = torch.randn(3, 5, K, K).double()
    W.requires_grad = True
    post = F.conv2d(x1, weight=W, stride=stride)
    x2 = F.relu(post)
    avg = x2.sum() / B
    avg.backward()
    target_grad = W.grad
    post = post.detach()
    x2 = x2.detach()
    pred_grad = vnn.convolutional_outer_product(x1, (post > 0.).double(),
        kernel_size=K, stride=stride).mean(dim=0)
    assert_allclose(target_grad, pred_grad)

@pytest.mark.parametrize("stride, padding, kernel_size", [(1, 0, 3), (1, 5, 5), (2, 3, 7), (4, 4, 2)])
def test_conv_outer_prod_grad_tanh_stride_padding_kernel(stride, padding, kernel_size):
    B = 128
    K = kernel_size
    x1 = torch.randn(B, 5, 36, 36).double()
    W = torch.randn(3, 5, K, K).double()
    W.requires_grad = True
    post = F.conv2d(x1, weight=W, stride=stride, padding=padding)
    x2 = torch.tanh(post)
    avg = x2.sum() / B
    avg.backward()
    target_grad = W.grad
    post = post.detach()
    x2 = x2.detach()
    pred_grad = vnn.convolutional_outer_product(x1, 1.-torch.tanh(post)**2,
        kernel_size=K, stride=stride, padding=padding).mean(dim=0)
    assert_allclose(target_grad, pred_grad)


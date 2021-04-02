import sys
sys.path.append("..")
import vnn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.testing import assert_allclose
import pytest 

@pytest.mark.parametrize("kernel_size, stride, padding, fc_size", [(5, 1, 0, 2352), (5, 2, 3, 867), (5, 4, 2, 192), (3, 7, 2, 75)])
def test_conv_grad(kernel_size, stride, padding, fc_size):
    net = nn.Sequential(vnn.Conv2d(category_dim=10, in_channels=3, out_channels=3, kernel_size=kernel_size, stride=stride, padding=padding),
                        vnn.ctReLU(),
                        vnn.Flatten(),
                        vnn.Linear(10, fc_size, 1))
    with torch.no_grad():
        net[-1].weight[:] = 1.
    x = torch.randn(16, 10, 3, 32, 32)
    out = net(x)[..., 0]
    loss = -F.log_softmax(out, dim=1)[:, 0].mean(dim=0)
    loss.backward()
    W_grad, b_grad = net[0].weight.grad.clone().numpy(), net[0].bias.grad.clone().numpy()
    for p in net.parameters():
        p.grad = None
    labels = torch.zeros(len(x)).long()
    vnn.set_model_grads(net, out, labels)
    W_update, b_update = net[0].weight.grad.clone().numpy(), net[0].bias.grad.clone().numpy()
    assert_allclose(W_update, W_grad, atol=1e-3)
    assert_allclose(b_update, b_grad, atol=1e-3)

#@pytest.mark.parametrize("kernel_size, stride, padding, fc_size", [(5, 1, 0, 2352), (5, 2, 3, 867), (5, 4, 2, 192), (3, 7, 2, 75)])
def test_linear_grad():
    net = nn.Sequential(vnn.Linear(category_dim=10, in_features=60, out_features=50),
                        vnn.tReLU(),
                        vnn.Linear(10, 50, 1))
    with torch.no_grad():
        net[-1].weight[:] = 1.
    x = torch.randn(16, 10, 60)
    out = net(x)[..., 0]
    loss = -F.log_softmax(out, dim=1)[:, 0].mean(dim=0)
    loss.backward()
    W_grad, b_grad = net[0].weight.grad.clone().numpy(), net[0].bias.grad.clone().numpy()
    for p in net.parameters():
        p.grad = None
    labels = torch.zeros(len(x)).long()
    vnn.set_model_grads(net, out, labels)
    W_update, b_update = net[0].weight.grad.clone().numpy(), net[0].bias.grad.clone().numpy()
    assert_allclose(W_update, W_grad, atol=1e-3)
    assert_allclose(b_update, b_grad, atol=1e-3)



@pytest.mark.parametrize("out_channels, kernel_size", [(4, 5), (4, 7), (8, 3)])
def test_lc_grad_1(out_channels, kernel_size):
    net = nn.Sequential(vnn.VecLocal2d(category_dim=10, in_channels=3, out_channels=out_channels,
                                       kernel_size=kernel_size, h_in=32, w_in=32, stride=1, padding=(kernel_size-1)//2),
                        vnn.ctReLU(),
                        vnn.Flatten(),
                        vnn.Linear(10, 4096*(out_channels // 4), 1))
    with torch.no_grad():
        net[-1].weight[:] = 1.
    x = torch.randn(16, 10, 3, 32, 32)
    out = net(x)[..., 0]
    loss = -F.log_softmax(out, dim=1)[:, 0].mean(dim=0)
    loss.backward()
    W_grad, b_grad = net[0].weight.grad.clone().numpy(), net[0].bias.grad.clone().numpy()
    for p in net.parameters():
        p.grad = None
    labels = torch.zeros(len(x)).long()
    vnn.set_model_grads(net, out, labels)
    W_update, b_update = net[0].weight.grad.clone().numpy(), net[0].bias.grad.clone().numpy()
    assert_allclose(W_update, W_grad, atol=1e-3)
    assert_allclose(b_update, b_grad, atol=1e-3)

@pytest.mark.parametrize("out_channels, kernel_size", [(4, 5), (8, 7)])
def test_lc_grad_2(out_channels, kernel_size):
    net = nn.Sequential(vnn.VecLocal2d(category_dim=13, in_channels=3, out_channels=out_channels,
                                       kernel_size=kernel_size, h_in=32, w_in=32, stride=1, padding=(kernel_size-1)//2),
                        vnn.ctReLU(),
                        vnn.Flatten(),
                        vnn.Linear(13, 4096*(out_channels // 4), 1))
    with torch.no_grad():
        net[-1].weight[:] = 1.
    x = torch.randn(16, 13, 3, 32, 32)
    out = net(x)[..., 0]
    loss = -F.log_softmax(out, dim=1)[:, 0].mean(dim=0)
    loss.backward()
    W_grad, b_grad = net[0].weight.grad.clone().numpy(), net[0].bias.grad.clone().numpy()
    for p in net.parameters():
        p.grad = None
    labels = torch.zeros(len(x)).long()
    vnn.set_model_grads(net, out, labels)
    W_update, b_update = net[0].weight.grad.clone().numpy(), net[0].bias.grad.clone().numpy()
    assert_allclose(W_update, W_grad, atol=1e-3)
    assert_allclose(b_update, b_grad, atol=1e-3)

@pytest.mark.parametrize("kernel_size", [2, 4, 7])
def test_lc_grad_3(kernel_size):
    net = nn.Sequential(vnn.VecLocal2d(category_dim=15, in_channels=4, out_channels=2,
                                       kernel_size=kernel_size, h_in=32, w_in=32, stride=1, padding=0),
                        vnn.ctReLU(),
                        vnn.Flatten(),
                        vnn.Linear(15, 2*(32-kernel_size+1)**2, 1))
    with torch.no_grad():
        net[-1].weight[:] = 1.
    x = torch.randn(16, 15, 4, 32, 32)
    out = net(x)[..., 0]
    loss = -F.log_softmax(out, dim=1)[:, 0].mean(dim=0)
    loss.backward()
    W_grad, b_grad = net[0].weight.grad.clone().numpy(), net[0].bias.grad.clone().numpy()
    for p in net.parameters():
        p.grad = None
    labels = torch.zeros(len(x)).long()
    vnn.set_model_grads(net, out, labels)
    W_update, b_update = net[0].weight.grad.clone().numpy(), net[0].bias.grad.clone().numpy()
    assert_allclose(W_update, W_grad, atol=1e-3)
    assert_allclose(b_update, b_grad, atol=1e-3)

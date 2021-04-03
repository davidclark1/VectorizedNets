import sys
sys.path.append("..")
import vnn
sys.path.append("../experiments")
import models
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.testing import assert_allclose
import pytest 

@pytest.mark.parametrize("mono", [True, False])
def test_mnist_fc(mono):
    x = torch.randn(16, 28*28)
    y = torch.randint(0, 10, (16,))
    input = vnn.expand_input(x, 10)
    model = models.make_mnist_vec_fc(mono)
    if mono:
        assert(model[2].weight.min().item() >= 0.)
        assert(model[4].weight.min().item() >= 0.)
    out = model(input)[..., 0]
    n1, n2 = vnn.set_model_grads(model, out, y)
    assert(n1 == 2)
    assert(n2 == 1)
    for (name, p) in model.named_parameters():
        if name[-2:] != ".t":
            assert(hasattr(p, "grad"))
            assert(p.grad.shape == p.shape)
            assert(p.grad.max().item() > 1e-4)
    params = list(model.parameters())
    assert(len(params) == 8)

@pytest.mark.parametrize("mono", [True, False])
def test_cifar_fc(mono):
    x = torch.randn(16, 3*32*32)
    y = torch.randint(0, 10, (16,))
    input = vnn.expand_input(x, 10)
    model = models.make_cifar_vec_fc(mono)
    if mono:
        assert(model[2].weight.min().item() >= 0.)
        assert(model[4].weight.min().item() >= 0.)
        assert(model[6].weight.min().item() >= 0.)
        assert(model[8].weight.min().item() >= 0.)
    out = model(input)[..., 0]
    n1, n2 = vnn.set_model_grads(model, out, y)
    assert(n1 == 4)
    assert(n2 == 1)
    for (name, p) in model.named_parameters():
        if name[-2:] != ".t":
            assert(hasattr(p, "grad"))
            assert(p.grad.shape == p.shape)
            assert(p.grad.max().item() > 1e-4)
    params = list(model.parameters())
    assert(len(params) == 14)

@pytest.mark.parametrize("mono", [True, False])
def test_mnist_conv(mono):
    x = torch.randn(16, 1, 28, 28)
    y = torch.randint(0, 10, (16,))
    input = vnn.expand_input_conv(x, 10)
    model = models.make_mnist_vec_conv(mono)
    if mono:
        assert(model[3].weight.min().item() >= 0.)
        assert(model[7].weight.min().item() >= 0.)
        assert(model[9].weight.min().item() >= 0.)
    out = model(input)[..., 0]
    n1, n2 = vnn.set_model_grads(model, out, y)
    assert(n1 == 3)
    assert(n2 == 1)
    for (name, p) in model.named_parameters():
        if name[-2:] != ".t":
            assert(hasattr(p, "grad"))
            assert(p.grad.shape == p.shape)
            assert(p.grad.max().item() > 1e-4)
    params = list(model.parameters())
    assert(len(params) == 11)

@pytest.mark.parametrize("mono", [True, False])
def test_cifar_conv(mono):
    x = torch.randn(16, 3, 32, 32)
    y = torch.randint(0, 10, (16,))
    input = vnn.expand_input_conv(x, 10)
    model = models.make_cifar_vec_conv(mono)
    if mono:
        assert(model[3].weight.min().item() >= 0.)
        assert(model[6].weight.min().item() >= 0.)
        assert(model[9].weight.min().item() >= 0.)
        assert(model[11].weight.min().item() >= 0.)
    out = model(input)[..., 0]
    n1, n2 = vnn.set_model_grads(model, out, y)
    assert(n1 == 4)
    assert(n2 == 1)
    for (name, p) in model.named_parameters():
        if name[-2:] != ".t":
            assert(hasattr(p, "grad"))
            assert(p.grad.shape == p.shape)
            assert(p.grad.max().item() > 1e-4)
    params = list(model.parameters())
    assert(len(params) == 14)

@pytest.mark.parametrize("mono", [True, False])
def test_mnist_lc(mono):
    x = torch.randn(16, 1, 28, 28)
    y = torch.randint(0, 10, (16,))
    input = vnn.expand_input_conv(x, 10)
    model = models.make_mnist_vec_lc(mono)
    if mono:
        assert(model[3].weight.min().item() >= 0.)
        assert(model[7].weight.min().item() >= 0.)
        assert(model[9].weight.min().item() >= 0.)
    out = model(input)[..., 0]
    n1, n2 = vnn.set_model_grads(model, out, y)
    assert(n1 == 3)
    assert(n2 == 1)
    for (name, p) in model.named_parameters():
        if name[-2:] != ".t":
            assert(hasattr(p, "grad"))
            assert(p.grad.shape == p.shape)
            assert(p.grad.max().item() > 1e-4)
    params = list(model.parameters())
    assert(len(params) == 11)

@pytest.mark.parametrize("mono", [True, False])
def test_cifar_lc(mono):
    x = torch.randn(16, 3, 32, 32)
    y = torch.randint(0, 10, (16,))
    input = vnn.expand_input_conv(x, 10)
    model = models.make_cifar_vec_lc(mono)
    if mono:
        assert(model[3].weight.min().item() >= 0.)
        assert(model[6].weight.min().item() >= 0.)
        assert(model[9].weight.min().item() >= 0.)
        assert(model[11].weight.min().item() >= 0.)
    out = model(input)[..., 0]
    n1, n2 = vnn.set_model_grads(model, out, y)
    assert(n1 == 4)
    assert(n2 == 1)
    for (name, p) in model.named_parameters():
        if name[-2:] != ".t":
            print(name)
            assert(hasattr(p, "grad"))
            assert(p.grad.shape == p.shape)
            assert(p.grad.max().item() > 1e-4)
    params = list(model.parameters())
    assert(len(params) == 14)

@pytest.mark.parametrize("mono", [True, False])
def test_post_step_callback_conv(mono):
    x = torch.randn(12, 3, 32, 32)
    y = torch.randint(0, 10, (12,))
    input = vnn.expand_input_conv(x, 10)
    model = models.make_cifar_vec_conv(mono)
    opt = torch.optim.SGD(model.parameters(), lr=1.)
    opt.zero_grad()
    out = model(input)[..., 0]
    vnn.set_model_grads(model, out, y)
    opt.step()
    assert(model[3].weight.min().item() <= 0.)
    assert(model[6].weight.min().item() <= 0.)
    assert(model[9].weight.min().item() <= 0.)
    assert(model[11].weight.min().item() <= 0.)
    vnn.post_step_callback(model)
    if mono:
        assert(model[3].weight.min().item() >= 0.)
        assert(model[6].weight.min().item() >= 0.)
        assert(model[9].weight.min().item() >= 0.)
        assert(model[11].weight.min().item() >= 0.)

@pytest.mark.parametrize("mono", [True, False])
def test_post_step_callback_lc(mono):
    x = torch.randn(16, 1, 28, 28)
    y = torch.randint(0, 10, (16,))
    input = vnn.expand_input_conv(x, 10)
    model = models.make_mnist_vec_lc(mono)
    opt = torch.optim.SGD(model.parameters(), lr=1.)
    opt.zero_grad()
    out = model(input)[..., 0]
    vnn.set_model_grads(model, out, y)
    opt.step()
    assert(model[3].weight.min().item() <= 0.)
    assert(model[7].weight.min().item() <= 0.)
    assert(model[9].weight.min().item() <= 0.)
    vnn.post_step_callback(model)
    if mono:
        assert(model[3].weight.min().item() >= 0.)
        assert(model[7].weight.min().item() >= 0.)
        assert(model[9].weight.min().item() >= 0.)






import sys
sys.path.append("..")
import vnn
import local2d
sys.path.append("../experiments")
import models
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.testing import assert_allclose
import pytest
from scipy.stats import pearsonr

DEVICE = 0

"""
    nonvec_models = [
        models.make_mnist_nonvec_fc(),
        models.make_mnist_nonvec_conv(),
        models.make_mnist_nonvec_lc()
    ]
"""

def compare_grads(model, input):
    labels = torch.randint(0, 10, (len(input),)).to(DEVICE)
    #custom backwards pass
    output = model(input)[..., 0]
    vnn.set_model_grads(model, output, labels, learning_rule="bp", reduction="mean")
    g1 = [param.grad.clone() for (name, param) in model.named_parameters() if name[-2:] != ".t"]
    #autograd backwards pass
    loss_fn = nn.CrossEntropyLoss(reduction="mean")
    vnn.zero_grads(model)
    vnn.toggle_grads(model, True)
    output = model(input)[..., 0]
    loss = loss_fn(output, labels)
    loss.backward()
    g2 = [param.grad.clone() for (name, param) in model.named_parameters() if name[-2:] != ".t"]
    for i in range(len(g1)):
        r = pearsonr(g1[i].cpu().flatten().numpy(), g2[i].cpu().flatten().numpy())[0]
        assert(r > 0.99999)

@pytest.mark.parametrize("model_idx", [0, 1])
def test_mnist_vec_fc_models(model_idx):
    vec_model_fns = [
        lambda: models.make_mnist_vec_fc(True),
        lambda: models.make_mnist_vec_fc(False),
    ]
    model = vec_model_fns[model_idx]().to(DEVICE)
    input = torch.randn(4, 10, 10*28*28).to(DEVICE)
    compare_grads(model, input)
    
@pytest.mark.parametrize("model_idx", [0, 1])
def test_mnist_vec_conv_models(model_idx):
    vec_model_fns = [
        lambda: models.make_mnist_vec_conv(True),
        lambda: models.make_mnist_vec_conv(False),
    ]
    model = vec_model_fns[model_idx]().to(DEVICE)
    input = torch.randn(4, 10, 10, 28, 28).to(DEVICE)
    compare_grads(model, input)

@pytest.mark.parametrize("model_idx", [0, 1])
def dont_test_mnist_vec_lc_models(model_idx):
    vec_model_fns = [
        lambda: models.make_mnist_vec_lc(True),
        lambda: models.make_mnist_vec_lc(False),
    ]
    model = vec_model_fns[model_idx]().to(DEVICE)
    input = torch.randn(4, 10, 10, 28, 28).to(DEVICE)
    compare_grads(model, input)

@pytest.mark.parametrize("model_idx", [0, 1])
def test_cifar_vec_fc_models(model_idx):
    vec_model_fns = [
        lambda: models.make_cifar_vec_fc(True),
        lambda: models.make_cifar_vec_fc(False),
    ]
    model = vec_model_fns[model_idx]().to(DEVICE)
    input = torch.randn(4, 10, 10*3*32*32).to(DEVICE)
    compare_grads(model, input)
    
@pytest.mark.parametrize("model_idx", [0, 1])
def test_cifar_vec_conv_models(model_idx):
    vec_model_fns = [
        lambda: models.make_cifar_vec_conv(True),
        lambda: models.make_cifar_vec_conv(False),
    ]
    model = vec_model_fns[model_idx]().to(DEVICE)
    input = torch.randn(4, 10, 30, 32, 32).to(DEVICE)
    compare_grads(model, input)

@pytest.mark.parametrize("model_idx", [0, 1])
def dont_test_cifar_vec_lc_models(model_idx):
    vec_model_fns = [
        lambda: models.make_cifar_vec_lc(True),
        lambda: models.make_cifar_vec_lc(False),
    ]
    model = vec_model_fns[model_idx]().to(DEVICE)
    input = torch.randn(4, 10, 30, 32, 32).to(DEVICE)
    compare_grads(model, input)
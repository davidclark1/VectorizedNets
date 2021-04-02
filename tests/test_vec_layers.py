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
    assert_allclose(out, target_out, rtol=1e-4)


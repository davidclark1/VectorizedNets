import sys
sys.path.append("..")
import vnn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.testing import assert_allclose
import pytest 


@pytest.mark.parametrize("category_dim, num_channels, dim", [(1, 3, 13), (3, 2, 18), (13, 4, 28)])
def test_flatten(category_dim, num_channels, dim):
    x = torch.randn(128, category_dim, num_channels, dim, dim)
    fla = vnn.Flatten()
    y = fla(x)
    assert(y.shape == (128, category_dim, num_channels*dim**2))

def test_flatten_order():
    x = torch.zeros(128, 10, 5, 16, 16)
    for i in range(5):
        x[:, :, i] = i
    fla = vnn.Flatten()
    y = fla(x).numpy()
    assert(np.all(y[:, :, 0::5] == 0))
    assert(np.all(y[:, :, 1::5] == 1))
    assert(np.all(y[:, :, 2::5] == 2))
    assert(np.all(y[:, :, 3::5] == 3))
    assert(np.all(y[:, :, 4::5] == 4))

def test_avg_pool():
    x = torch.zeros(128, 13, 5, 16, 16)
    pool = vnn.AvgPool2d(4)
    y = pool(x)
    assert(y.shape == (128, 13, 5, 4, 4))
    target = torch.zeros(128, 13, 5, 4, 4)
    for i in range(13):
        target[:, i] = F.avg_pool2d(x[:, i], 4)
    assert_allclose(target, y)

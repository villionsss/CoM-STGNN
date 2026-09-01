"""Chebyshev graph convolution used by the main CoM-STGNN model.

The kernel-generation helpers are retained for reproducibility, but the
public workflow loads the prepared kernels included in ``data/`` directly.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


def calculate_scaled_laplacian(adj):
    """Compute the scaled normalized Laplacian for a weighted graph."""
    adjacency = np.asarray(adj, dtype=np.float32)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError('adj must be a square matrix')

    node_count = adjacency.shape[0]
    degree = np.sum(adjacency, axis=1)
    laplacian = np.diag(degree) - adjacency

    for row in range(node_count):
        for column in range(node_count):
            if degree[row] > 0 and degree[column] > 0:
                laplacian[row, column] /= np.sqrt(
                    degree[row] * degree[column]
                )

    laplacian[~np.isfinite(laplacian)] = 0
    largest_eigenvalue = np.linalg.eigvals(laplacian).max().real
    if largest_eigenvalue <= 0:
        raise ValueError('The graph Laplacian must have a positive eigenvalue')

    return (
        2 * laplacian / largest_eigenvalue
        - np.eye(node_count, dtype=laplacian.dtype)
    )


def calculate_cheb_poly(laplacian, ks):
    """Generate the first ``ks`` Chebyshev polynomials of a Laplacian."""
    if ks < 1:
        raise ValueError('ks must be at least 1')

    scaled_laplacian = np.asarray(laplacian, dtype=np.float32)
    if (
        scaled_laplacian.ndim != 2
        or scaled_laplacian.shape[0] != scaled_laplacian.shape[1]
    ):
        raise ValueError('laplacian must be a square matrix')

    node_count = scaled_laplacian.shape[0]
    polynomials = [
        np.eye(node_count, dtype=scaled_laplacian.dtype),
        scaled_laplacian.copy(),
    ]
    for _ in range(2, ks):
        polynomials.append(
            2 * np.matmul(scaled_laplacian, polynomials[-1])
            - polynomials[-2]
        )

    return np.asarray(polynomials[:ks])


class Align(nn.Module):
    def __init__(self, c_in, c_out):
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        if c_in > c_out:
            self.conv1x1 = nn.Conv2d(c_in, c_out, 1)

    def forward(self, x):
        if self.c_in > self.c_out:
            return self.conv1x1(x)
        if self.c_in < self.c_out:
            return F.pad(x, [0, 0, 0, 0, 0, self.c_out - self.c_in, 0, 0])
        return x


class GraphConvolution(nn.Module):
    def __init__(self, config, lk, in_channels=16, out_channels=16):
        super(GraphConvolution, self).__init__()

        kernels = torch.as_tensor(lk, dtype=torch.float32)
        if kernels.ndim != 3:
            raise ValueError('lk must have shape (Ks, num_nodes, num_nodes)')
        if kernels.shape[0] != config.Ks:
            raise ValueError('The first dimension of lk must equal config.Ks')
        if kernels.shape[1] != kernels.shape[2]:
            raise ValueError('lk must be square in the node dimensions')

        self.register_buffer('Lk', kernels.contiguous())
        self.theta = nn.Parameter(
            torch.FloatTensor(in_channels, out_channels, config.Ks).to(config.device)
        )
        self.b = nn.Parameter(
            torch.FloatTensor(1, out_channels, 1, 1).to(config.device)
        )
        self.align = Align(in_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.theta)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.b, -bound, bound)

    def forward(self, x):
        # x: (batch, channels, nodes, time)
        if x.ndim != 4:
            raise ValueError('x must have shape (batch, channels, nodes, time)')
        if x.shape[2] != self.Lk.shape[1]:
            raise ValueError('x and Lk must use the same number of nodes')

        x = x.permute(0, 1, 3, 2)
        x_c = torch.einsum('knm,bctm->bctkn', self.Lk, x)
        x_gc = torch.einsum('iok,bctkn->botn', self.theta, x_c) + self.b
        x_in = self.align(x)
        x_out = torch.relu(x_gc + x_in)
        return x_out.permute(0, 1, 3, 2)

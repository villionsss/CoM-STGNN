"""Temporal gated convolution and graph convolution used by CoM-STGNN."""

import torch
from torch import nn
import torch.nn.functional as F

from model import Layer_GCN


class TGCN_M(nn.Module):
    def __init__(self, config, graph, emd_dim, hide_dim, out_dim):
        super(TGCN_M, self).__init__()

        del emd_dim

        self.num_nodes = config.num_of_vertices
        self.feature_dim = 1
        self.kernel_size = config.Kt
        self.blocks = 1
        self.layers = 2

        self.nhid = hide_dim
        self.residual_channels = self.nhid
        self.dilation_channels = self.nhid
        self.skip_channels = self.nhid * 2
        self.end_channels = self.nhid * 4
        self.input_window = config.input_window
        self.output_window = out_dim
        self.output_dim = 1

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.start_conv = nn.Conv2d(
            in_channels=self.feature_dim,
            out_channels=self.residual_channels,
            kernel_size=(1, 1),
        )

        receptive_field = self.output_dim
        self.dropout = nn.Dropout(p=0.2)

        for _ in range(self.blocks):
            additional_scope = self.kernel_size - 1
            new_dilation = 1
            for _ in range(self.layers):
                temporal_kernel = (1, self.kernel_size)
                self.filter_convs.append(
                    nn.Conv2d(
                        in_channels=self.residual_channels,
                        out_channels=self.dilation_channels,
                        kernel_size=temporal_kernel,
                        dilation=(1, new_dilation),
                    )
                )
                # Conv2d applies the same temporal operation independently at
                # every node, matching the intended four-dimensional layout.
                self.gate_convs.append(
                    nn.Conv2d(
                        in_channels=self.residual_channels,
                        out_channels=self.dilation_channels,
                        kernel_size=temporal_kernel,
                        dilation=(1, new_dilation),
                    )
                )
                self.skip_convs.append(
                    nn.Conv2d(
                        in_channels=self.dilation_channels,
                        out_channels=self.skip_channels,
                        kernel_size=(1, 1),
                    )
                )
                self.bn.append(nn.BatchNorm2d(self.residual_channels))
                self.gconv.append(
                    Layer_GCN.GraphConvolution(
                        config,
                        graph,
                        self.dilation_channels,
                        self.residual_channels,
                    )
                )
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2

        self.end_conv_1 = nn.Conv2d(
            in_channels=self.skip_channels,
            out_channels=self.end_channels,
            kernel_size=(1, 1),
            bias=True,
        )
        self.end_conv_2 = nn.Conv2d(
            in_channels=self.end_channels,
            out_channels=self.output_window,
            kernel_size=(1, 1),
            bias=True,
        )
        self.receptive_field = receptive_field

    def forward(self, flow):
        # flow: (batch, time, nodes, feature)
        inputs = flow.transpose(1, 3)
        inputs = F.pad(inputs, (1, 0, 0, 0))
        x = self.start_conv(inputs)
        skip = None

        for i in range(self.blocks * self.layers):
            residual = x
            filter_output = torch.tanh(self.filter_convs[i](residual))
            gate_output = torch.sigmoid(self.gate_convs[i](residual))
            x = filter_output * gate_output

            skip_output = self.skip_convs[i](x)
            if skip is not None:
                skip = skip[:, :, :, -skip_output.size(3):]
                skip = skip_output + skip
            else:
                skip = skip_output

            x = self.gconv[i](x)
            x = self.dropout(x)
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)

        batch, channels, nodes, steps = x.shape
        return x.view(batch, channels * steps, nodes).permute(0, 2, 1)

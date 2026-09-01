"""Self-attention layer for cross-modal feature fusion."""

import torch
from torch import nn


class SelfAttention(nn.Module):
    def __init__(self, dim_out, num_views):
        super(SelfAttention, self).__init__()
        self.dim_out = dim_out
        self.att_dim = dim_out * num_views

        self.q_conv = nn.Conv1d(
            in_channels=self.att_dim,
            out_channels=self.att_dim // 2,
            kernel_size=1,
        )
        self.k_conv = nn.Conv1d(
            in_channels=self.att_dim,
            out_channels=self.att_dim // 2,
            kernel_size=1,
        )
        self.v_conv = nn.Conv1d(
            in_channels=self.att_dim,
            out_channels=self.att_dim // 2,
            kernel_size=1,
        )

        self.out_linear = nn.Linear(self.att_dim // 2, dim_out)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

    def forward(self, feature_views):
        if len(feature_views) > 1:
            features = torch.cat(feature_views, dim=-1)
        else:
            features = feature_views[0]

        _, _, feature_dim = features.size()
        features = features.transpose(1, 2)

        query = self.q_conv(features)
        key = self.k_conv(features)
        value = self.v_conv(features)

        attention_weights = torch.matmul(
            query, key.transpose(-2, -1)
        ) / (feature_dim ** 0.5)
        attention_weights = self.softmax(attention_weights)

        attended = torch.matmul(attention_weights, value)
        attended = attended.transpose(1, 2)
        attended = self.out_linear(attended)
        attended = self.relu(attended)
        return self.dropout(attended)

import torch
import torch.nn as nn


class SequenceChannelGate(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        hidden_channels = max(1, channels // reduction_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1)
        max_pool, _ = torch.max(x, dim=1)
        scale = self.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool)).unsqueeze(1)
        return x * scale


class SequenceSpatialGate(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        seq_first = x.transpose(1, 2)
        avg_pool = torch.mean(seq_first, dim=1, keepdim=True)
        max_pool, _ = torch.max(seq_first, dim=1, keepdim=True)
        scale = self.sigmoid(self.conv(torch.cat([avg_pool, max_pool], dim=1))).transpose(1, 2)
        return x * scale


class SequenceCBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16, spatial_kernel_size=7):
        super().__init__()
        self.channel_gate = SequenceChannelGate(channels, reduction_ratio=reduction_ratio)
        self.spatial_gate = SequenceSpatialGate(kernel_size=spatial_kernel_size)

    def forward(self, x):
        x = self.channel_gate(x)
        x = self.spatial_gate(x)
        return x

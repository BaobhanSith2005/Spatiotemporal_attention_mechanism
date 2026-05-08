import csv

import torch
import torch.nn as nn
import torch.nn.functional as F

from Spatiotemporal_attention_mechanism.spatial_attention.CBAM import CBAM
from Spatiotemporal_attention_mechanism.temporal_attention.multihead_self_attention import MultiHeadSelfAttention
from python_scripts.PPO.PPO_PPOnet_attention4_1 import (
    ActorCritic as FourOneActorCritic,
    BidirectionalCrossAttentionBlock,
)


class ImageSpatialAttentionEncoder(nn.Module):
    """4_1-compatible image encoder with spatial attention, output shape [B, 1, 32]."""

    def __init__(self, output_dim=32, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.cbam1 = CBAM(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.cbam2 = CBAM(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.cbam3 = CBAM(64)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.cbam4 = CBAM(64)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.flatten = nn.Flatten()
        self.head = nn.Linear(64 * 8 * 8, output_dim)

    def forward(self, img):
        x = self.conv1(img)
        x = self.cbam1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.cbam2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.cbam3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv4(x)
        x = self.cbam4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.pool(x)
        x = self.flatten(x)
        x = self.head(x)
        x_min = x.min(dim=1, keepdim=True)[0]
        x_max = x.max(dim=1, keepdim=True)[0]
        x = (x - x_min) / (x_max - x_min + 1e-8)
        return x.unsqueeze(1)


class StateGraphSpatialAttentionEncoder(nn.Module):
    """4_1-compatible angle-graph spatial-attention encoder, output shape [B, 4, 32]."""

    def __init__(self, output_dim=32, dropout=0.1):
        super().__init__()
        import torch_geometric.nn as tgnn

        self.conv1 = tgnn.SAGEConv(1, 16, normalize=True)
        self.bn1 = tgnn.BatchNorm(16)
        self.conv2 = tgnn.GATConv(16, output_dim, heads=1)
        self.bn2 = tgnn.BatchNorm(output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward_single(self, graph):
        x = graph.x
        edge_index = graph.edge_index

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        mn = x.min()
        mx = x.max()
        return (x - mn) / (mx - mn + 1e-8)

    def forward(self, graph_result):
        if isinstance(graph_result, list):
            batch_graph_feats = [self.forward_single(graph).unsqueeze(0) for graph in graph_result]
            return torch.cat(batch_graph_feats, dim=0)
        return self.forward_single(graph_result).unsqueeze(0)


class TemporalAttentionEncoder(nn.Module):
    """4_1-compatible temporal attention encoder, input [B, T, 32] -> output [B, 1, 32]."""

    def __init__(self, input_dim=32, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(input_dim, num_heads, dropout)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, sequence):
        output = self.attn(sequence)
        last_token = output[:, -1:, :]
        return self.norm(last_token)


class StrictFourOneVariantActorCriticBase(FourOneActorCritic):
    """
    Strict 4_1-based ablation base.
    Inherits all buffer logic, fusion order, forward path, and heads from 4_1,
    and only swaps the four experiment-controlled branches.
    """

    image_spatial_mode = "conv"
    state_spatial_mode = "gnn"
    image_temporal_mode = "rnn"
    state_temporal_mode = "rnn"

    def __init__(self, act_dim):
        super().__init__(act_dim)

        if self.image_spatial_mode == "attention":
            self.image_spatial_encoder = ImageSpatialAttentionEncoder(output_dim=32, dropout=0.1)

        if self.state_spatial_mode == "attention":
            self.state_graph_encoder = StateGraphSpatialAttentionEncoder(output_dim=32, dropout=0.1)

        if self.image_temporal_mode == "attention":
            self.image_temporal_encoder = TemporalAttentionEncoder(input_dim=32, num_heads=4, dropout=0.1)

        if self.state_temporal_mode == "attention":
            self.state_temporal_encoder = TemporalAttentionEncoder(input_dim=32, num_heads=4, dropout=0.1)


class AttentionFusionAblationActorCriticBase(StrictFourOneVariantActorCriticBase):
    """
    All-attention ablation base with configurable fusion styles.

    The three fusion switches are:
    1. spatial_fusion_mode: image spatial branch <-> angle-graph spatial branch
    2. final_fusion_mode: spatial block <-> temporal block
    3. temporal_fusion_mode: image temporal branch <-> angle temporal branch
    """

    image_spatial_mode = "attention"
    state_spatial_mode = "attention"
    image_temporal_mode = "attention"
    state_temporal_mode = "attention"

    spatial_fusion_mode = "concat"
    final_fusion_mode = "concat"
    temporal_fusion_mode = "concat"

    def __init__(self, act_dim):
        super().__init__(act_dim)
        self.final_cross_attention = BidirectionalCrossAttentionBlock(dim=64, heads=1, dropout=0.1)
        self._validate_fusion_mode(self.spatial_fusion_mode, "spatial_fusion_mode")
        self._validate_fusion_mode(self.final_fusion_mode, "final_fusion_mode")
        self._validate_fusion_mode(self.temporal_fusion_mode, "temporal_fusion_mode")

    @staticmethod
    def _validate_fusion_mode(mode, attr_name):
        valid_modes = {"concat", "cross_attention"}
        if mode not in valid_modes:
            raise ValueError(f"{attr_name} must be one of {valid_modes}, got {mode!r}")

    @staticmethod
    def _repeat_tokens_if_needed(source_tokens, target_tokens):
        if source_tokens.shape[1] == target_tokens.shape[1]:
            return source_tokens
        return source_tokens.repeat(1, target_tokens.shape[1], 1)

    def _fuse_spatial_features(self, image_spatial, state_spatial):
        if self.spatial_fusion_mode == "cross_attention":
            spatial_img_ctx, spatial_state_ctx = self.spatial_cross_attention(image_spatial, state_spatial)
        else:
            spatial_img_ctx, spatial_state_ctx = image_spatial, state_spatial

        spatial_img_ctx = self._repeat_tokens_if_needed(spatial_img_ctx, spatial_state_ctx)
        spatial_features = torch.cat([spatial_img_ctx, spatial_state_ctx], dim=-1)
        return self.spatial_projector(spatial_features)

    def _fuse_temporal_features(self, temporal_img_out, temporal_state_out):
        if self.temporal_fusion_mode == "cross_attention":
            temporal_img_ctx, temporal_state_ctx = self.temporal_cross_attention(
                temporal_img_out,
                temporal_state_out,
            )
        else:
            temporal_img_ctx, temporal_state_ctx = temporal_img_out, temporal_state_out

        temporal_features = torch.cat([temporal_img_ctx, temporal_state_ctx], dim=-1)
        return self.temporal_projector(temporal_features)

    def _fuse_final_features(self, spatial_features, temporal_features):
        temporal_summary = temporal_features.mean(dim=1, keepdim=True)
        spatial_summary = spatial_features.mean(dim=1, keepdim=True)

        if self.final_fusion_mode == "cross_attention":
            spatial_summary, temporal_summary = self.final_cross_attention(spatial_summary, temporal_summary)

        final_features = torch.cat([temporal_summary, spatial_summary], dim=-1)
        return self.final_projector(final_features)

    def forward(self, img, state):
        img, st_t = self._normalize_inputs(img, state)
        image_spatial = self.image_spatial_encoder(img)
        graph_result = self.creat_graph(st_t)
        state_spatial = self.state_graph_encoder(graph_result)

        spatial_features = self._fuse_spatial_features(image_spatial, state_spatial)

        state_temporal_seed = self.state_temporal_projector(st_t)
        temporal_img_out = self._encode_temporal_sequence(
            image_spatial,
            self.buffer_img,
            self.image_temporal_encoder,
            self.update_buffer_img,
        )
        temporal_state_out = self._encode_temporal_sequence(
            state_temporal_seed,
            self.buffer_state,
            self.state_temporal_encoder,
            self.update_buffer_state,
        )
        temporal_features = self._fuse_temporal_features(temporal_img_out, temporal_state_out)

        final_features = self._fuse_final_features(spatial_features, temporal_features)

        pooled = final_features.mean(dim=1)
        mu = self.mu_layer(pooled)
        raw_log_std = self.log_std_layer(pooled)
        raw_log_std = torch.clamp(raw_log_std, min=self.log_std_min * 2, max=self.log_std_max * 2)
        sigma = F.softplus(raw_log_std) + 1e-6

        mu = torch.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
        sigma = torch.nan_to_num(sigma, nan=1e-3, posinf=1e6, neginf=1e-6)

        value = self.critic(pooled)
        value = torch.nan_to_num(value, nan=0.0, posinf=1e6, neginf=-1e6)
        try:
            csv_file_path = "F:\\project_Spatiotemporal_attention_mechanism\\python_scripts\\PPO\\PPO_PPOnet_attention_new.csv"
            with open(csv_file_path, mode="a", newline="") as file:
                writer2 = csv.writer(file)
                writer2.writerow(
                    [
                        mu.detach().cpu().numpy().tolist(),
                        sigma.detach().cpu().numpy().tolist(),
                        value.detach().cpu().numpy().tolist(),
                    ]
                )
        except Exception:
            pass
        return mu, sigma, value

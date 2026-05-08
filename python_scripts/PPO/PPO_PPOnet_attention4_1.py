from collections import deque
import gc  # 移到文件顶部，避免重复导入
import random  # 用于ReplayBuffer的sample方法
import math
import numpy as np
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.data import Data
import torch.nn.functional as F
from python_scripts.Project_config import device
import csv
import os
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        # 使用deque的maxlen参数，当添加新元素且超出容量时，自动移除最旧的元素
        self.buffer = deque(maxlen=capacity)
        
    def push(self, transition_dict):
        # 直接添加，deque会自动处理容量限制
        self.buffer.append(transition_dict)
        
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return list(self.buffer)  # 如果缓冲区数据不足，返回所有数据
        return random.sample(list(self.buffer), batch_size)
        
    def clear(self):
        self.buffer.clear()
        
    def __len__(self):
        return len(self.buffer)


# 决策层
class DecisionLayer(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=1):
        super().__init__()
        # 节点级特征处理
        self.node_fc1 = nn.Linear(input_dim, hidden_dim)
        self.node_fc2 = nn.Linear(hidden_dim, hidden_dim)

        # 全局上下文处理
        self.global_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)

        # 决策输出层
        self.decision_fc = nn.Linear(hidden_dim, output_dim)

        # 层归一化
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # 激活函数
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        """x: [batch_size, num_nodes, feature_dim]"""
        # 1. 节点级特征提取
        x = self.relu(self.node_fc1(x))
        x = self.dropout(x)
        x = self.relu(self.node_fc2(x))
        x = self.norm1(x)

        # 2. 全局注意力机制（捕获节点间依赖）
        attn_out, _ = self.global_attn(x, x, x)
        x = x + attn_out  # 残差连接
        x = self.norm2(x)

        # 3. 决策输出
        logits = self.decision_fc(x)  # [batch_size, num_nodes, 1]
        logits = logits.squeeze(-1)  # [batch_size, num_nodes]

        return logits


class TokenProjector(nn.Module):
    """Project concatenated token features to a target dimension."""

    def __init__(self, input_dim, output_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class BidirectionalCrossAttentionBlock(nn.Module):
    """Decoupled bidirectional cross-attention block for feature interaction."""

    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.a_to_b = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.b_to_a = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_a1 = nn.LayerNorm(dim)
        self.norm_b1 = nn.LayerNorm(dim)

    def forward(self, a, b):
        a_attn, _ = self.a_to_b(a, b, b)
        b_attn, _ = self.b_to_a(b, a, a)
        a = self.norm_a1(a + a_attn)
        b = self.norm_b1(b + b_attn)
        return a, b


class ImageConvEncoder(nn.Module):
    """Pure image convolution encoder for spatial features."""

    def __init__(self, output_dim=32, dropout=0.1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool2d((8, 8)),
        )
        self.flatten = nn.Flatten()
        self.head = nn.Sequential(
            nn.Linear(8 * 8 * 8, output_dim),
        )

    def forward(self, img):
        feat = self.features(img)
        feat = self.flatten(feat)
        feat = self.head(feat)
        feat_min = feat.min(dim=1, keepdim=True)[0]
        feat_max = feat.max(dim=1, keepdim=True)[0]
        feat = (feat - feat_min) / (feat_max - feat_min + 1e-8)
        return feat.unsqueeze(1)


class StateGraphEncoder(nn.Module):
    """Graph encoder for 4-D angle state."""

    def __init__(self, output_dim=32, dropout=0.1):
        super().__init__()
        import torch_geometric.nn as tgnn

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv_graph1 = tgnn.GCNConv(1, 16)
        self.fc = nn.Linear(16, output_dim)

    def forward_single(self, graph):
        x = graph.x
        edge_index = graph.edge_index
        x = self.conv_graph1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc(x)
        mn = x.min()
        mx = x.max()
        return (x - mn) / (mx - mn + 1e-8)

    def forward(self, graph_result):
        if isinstance(graph_result, list):
            batch_graph_feats = [self.forward_single(graph).unsqueeze(0) for graph in graph_result]
            return torch.cat(batch_graph_feats, dim=0)
        return self.forward_single(graph_result).unsqueeze(0)


class StateSequenceProjector(nn.Module):
    """Project raw angle state to the shared temporal token space."""

    def __init__(self, input_dim=4, output_dim=32, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x).unsqueeze(1)


class TemporalRNNEncoder(nn.Module):
    """Temporal encoder based on GRU, easy to swap for LSTM/Transformer later."""

    def __init__(self, input_dim=32, hidden_dim=32, num_layers=1, dropout=0.0):
        super().__init__()
        self.rnn = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, sequence):
        output, _ = self.rnn(sequence)
        last_token = output[:, -1:, :]
        return self.norm(last_token)


class ActorCritic(nn.Module):
    def __init__(self, act_dim):
        super().__init__()
        self.device = device
        self.relu = nn.ReLU()
        self.buffer_img = deque(maxlen=20)      # each element: Tensor [1, 128]
        self.buffer_state = deque(maxlen=20)    # each element: Tensor [1, 128]
        shared_dim = 32
        fusion_dim = 64
        final_dim = 128

        # Simplified backbone:
        # (图像卷积 + 角度图GNN + 空间交叉注意力) + (图像RNN + 角度RNN + 时间交叉注意力) + PPO
        self.image_spatial_encoder = ImageConvEncoder(output_dim=shared_dim, dropout=0.1)
        self.state_graph_encoder = StateGraphEncoder(output_dim=shared_dim, dropout=0.1)
        self.state_temporal_projector = StateSequenceProjector(input_dim=4, output_dim=shared_dim, dropout=0.1)
        self.image_temporal_encoder = TemporalRNNEncoder(input_dim=shared_dim, hidden_dim=shared_dim)
        self.state_temporal_encoder = TemporalRNNEncoder(input_dim=shared_dim, hidden_dim=shared_dim)
        self.spatial_cross_attention = BidirectionalCrossAttentionBlock(dim=shared_dim, heads=1, dropout=0.1)
        self.temporal_cross_attention = BidirectionalCrossAttentionBlock(dim=shared_dim, heads=1, dropout=0.1)
        self.spatial_projector = TokenProjector(input_dim=fusion_dim, output_dim=fusion_dim, dropout=0.1)
        self.temporal_projector = TokenProjector(input_dim=fusion_dim, output_dim=fusion_dim, dropout=0.1)
        self.final_projector = TokenProjector(input_dim=final_dim, output_dim=final_dim, dropout=0.1)

        self.decision_layer = DecisionLayer(input_dim=final_dim, hidden_dim=64, output_dim=1)
        self.critic = nn.Linear(final_dim, 1)
        self.mu_layer = nn.Linear(final_dim, act_dim)
        self.log_std_layer = nn.Linear(final_dim, act_dim)
        self.log_std_min = -20
        self.log_std_max = 2

        nn.init.constant_(self.log_std_layer.weight, 0.0)
        nn.init.constant_(self.log_std_layer.bias, -1.0)

    def update_buffer_img(self, feature: torch.Tensor):
        if not isinstance(feature, torch.Tensor):
            raise TypeError("update_buffer_img expects torch.Tensor")
        f = feature.detach().cpu()
        if f.dim() == 3:
            f = f.squeeze(1)
        if f.dim() == 2 and f.shape[0] == 1:
            self.buffer_img.append(f.clone())
        elif f.dim() == 2:
            for i in range(f.shape[0]):
                self.buffer_img.append(f[i:i + 1].clone())
        else:
            raise ValueError(f"Unexpected image buffer shape: {tuple(f.shape)}")

    def update_buffer_state(self, feature: torch.Tensor):
        if not isinstance(feature, torch.Tensor):
            raise TypeError("update_buffer_state expects torch.Tensor")
        f = feature.detach().cpu()
        if f.dim() == 3:
            f = f.squeeze(1)
        if f.dim() == 2 and f.shape[0] == 1:
            self.buffer_state.append(f.clone())
        elif f.dim() == 2:
            for i in range(f.shape[0]):
                self.buffer_state.append(f[i:i + 1].clone())
        else:
            raise ValueError(f"Unexpected state buffer shape: {tuple(f.shape)}")

    def reset_buffer(self):
        self.buffer_img.clear()
        self.buffer_state.clear()

    def _normalize_inputs(self, img, state):
        if isinstance(state, list) or isinstance(state, np.ndarray):
            st_t = torch.from_numpy(np.asarray(state)).float()
        elif isinstance(state, torch.Tensor):
            st_t = state.float()
        else:
            st_t = torch.tensor(state, dtype=torch.float32)
        if st_t.dim() == 1:
            st_t = st_t.unsqueeze(0)  # (1, state_dim)
        state_batch = st_t.size(0)

        # img
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(np.asarray(img)).float()
        # Accept dims: (H,W), (C,H,W), (N,H,W), (N,C,H,W)
        if img.dim() == 2:
            img = img.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        elif img.dim() == 3:
            c0 = img.size(0)
            if c0 == 1:
                img = img.unsqueeze(0)  # (1,1,H,W)
            elif c0 == state_batch:
                # 与 state batch 一致时优先按 (N,H,W) 处理，避免把 N=3 误判成 RGB
                img = img.unsqueeze(1)  # (N,1,H,W)
            elif c0 == 3:
                # 单张 RGB 图像
                img = img.mean(dim=0, keepdim=True).unsqueeze(0)
            else:
                # 默认按批量灰度图处理
                img = img.unsqueeze(1)  # (N,1,H,W)
        elif img.dim() == 4:
            N, C, H, W = img.shape
            if C != 1:
                img = img.mean(dim=1, keepdim=True)  # (N,1,H,W)
        else:
            raise ValueError(f"Unsupported img dims: {img.shape}")

        # move to device
        img = img.to(self.device)
        st_t = st_t.to(self.device)
        return img, st_t

    def _stack_buffer(self, buffer_deque):
        if len(buffer_deque) == 0:
            return None
        seq = torch.cat(list(buffer_deque), dim=0)  # [T, D] on CPU
        return seq.unsqueeze(0).to(self.device)     # [1, T, D]

    def _encode_temporal_sequence(self, current_feature, buffer_deque, temporal_encoder, update_fn):
        batch_size = current_feature.shape[0]
        if batch_size == 1:
            update_fn(current_feature)
            seq = self._stack_buffer(buffer_deque)
            if seq is not None:
                return temporal_encoder(seq)
        return temporal_encoder(current_feature)

    def forward(self, img, state):
        img, st_t = self._normalize_inputs(img, state)
        image_spatial = self.image_spatial_encoder(img)  # [B, 1, 32]
        graph_result = self.creat_graph(st_t)
        state_spatial = self.state_graph_encoder(graph_result)  # [B, 4, 32]

        spatial_img_ctx, spatial_state_ctx = self.spatial_cross_attention(image_spatial, state_spatial)
        spatial_img_ctx = spatial_img_ctx.repeat(1, spatial_state_ctx.shape[1], 1)
        spatial_features = torch.cat([spatial_img_ctx, spatial_state_ctx], dim=-1)  # [B, 4, 256]
        spatial_features = self.spatial_projector(spatial_features)

        state_temporal_seed = self.state_temporal_projector(st_t)  # [B, 1, 32]
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
        temporal_img_ctx, temporal_state_ctx = self.temporal_cross_attention(
            temporal_img_out,
            temporal_state_out,
        )
        temporal_features = torch.cat([temporal_img_ctx, temporal_state_ctx], dim=-1)  # [B, 1, 64]
        temporal_features = self.temporal_projector(temporal_features)

        temporal_summary = temporal_features.mean(dim=1, keepdim=True)
        spatial_summary = spatial_features.mean(dim=1, keepdim=True)
        final_features = torch.cat([temporal_summary, spatial_summary], dim=-1)  # [B, 1, 128]
        final_features = self.final_projector(final_features)

        pooled = final_features.mean(dim=1)  # [B, 128]
        mu = self.mu_layer(pooled)
        raw_log_std = self.log_std_layer(pooled)
        raw_log_std = torch.clamp(raw_log_std, min=self.log_std_min * 2, max=self.log_std_max * 2)
        sigma = F.softplus(raw_log_std) + 1e-6

        mu = torch.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
        sigma = torch.nan_to_num(sigma, nan=1e-3, posinf=1e6, neginf=1e-6)

        value = self.critic(pooled)
        value = torch.nan_to_num(value, nan=0.0, posinf=1e6, neginf=-1e6)
        try:
            csv_file_path = 'F:\\project_Spatiotemporal_attention_mechanism\\python_scripts\\PPO\\PPO_PPOnet_attention_new.csv'
            with open(csv_file_path, mode="a", newline="") as file:
                writer2 = csv.writer(file)
                writer2.writerow([mu.detach().cpu().numpy().tolist(), sigma.detach().cpu().numpy().tolist(), value.detach().cpu().numpy().tolist()])
        except Exception:
            pass
        return mu, sigma, value
    def create_edge_index(self):
        """创建节点的完整边索引"""
        # 定义节点数量
        num_nodes = 4  

        # 创建链式结构的双向边
        edge_list = []
        for i in range(num_nodes - 1):
            # 添加双向边 (i -> i+1) 和 (i+1 -> i)
            edge_list.append([i, i + 1])
            edge_list.append([i + 1, i])

        # 转换为PyG格式 [2, num_edges]
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous().to(device)
        return edge_index

    def creat_x(self, x_graph):
        """创建节点特征张量，支持批量和单个输入"""
        # 处理批量输入
        if isinstance(x_graph, torch.Tensor):
            # 检查是否为批量输入
            if len(x_graph.shape) >= 2:
                # 批量形状: [batch_size, 4] -> [batch_size, 4, 1]
                return x_graph.clone().detach().to(torch.float32).unsqueeze(-1)
            else:
                # 单个输入: [4] -> [4, 1]
                return x_graph.clone().detach().to(torch.float32).view(-1, 1)
        else:
            # 处理非张量输入
            x_tensor = torch.tensor(x_graph, dtype=torch.float32)
            if len(x_tensor.shape) >= 2:
                # 批量输入
                return x_tensor.view(-1, 4, 1).to(device)
            else:
                # 单个输入
                return x_tensor.view(-1, 1).to(device)

    def creat_graph(self, x_graph):
        """创建包含节点的图结构，支持批量和单个输入"""
        # 创建边索引
        edge_index = self.create_edge_index()
        
        # 创建节点特征
        x = self.creat_x(x_graph)
        
        # 检查是否为批量输入
        if len(x.shape) == 3:  # 批量输入形状: [batch_size, 4, 1]
            batch_size = x.shape[0]
            # 为每个样本创建图
            graphs = []
            for i in range(batch_size):
                # 创建单个图
                graph = Data(x=x[i], edge_index=edge_index)
                graph.x = graph.x.to(device)
                graphs.append(graph)
            
            # 如果使用PyG，可以返回图列表或者使用Batch.from_data_list
            # 这里返回图列表，让调用方决定如何处理
            return graphs
        else:  # 单个输入
            # 创建单个图数据
            graph = Data(x=x, edge_index=edge_index)
            # 确保在正确的设备上
            graph.x = graph.x.to(device)
            
            return graph


class PPO:
    def __init__(self, policy: torch.nn.Module, act_dim: int,
                 lr=1e-4, clip_ratio=0.2, update_epochs=10, minibatch_size=64,
                 gamma=0.99, lam=0.95, entropy_coef=0.2, value_coef=0.5,  # 增加默认entropy_coef到0.2
                 max_grad_norm=0.5, device=device):
        """
        :param policy: ActorCritic 网络（新策略），必须返回 mu, sigma, value
        :param act_dim: 动作维度（标量动作取 1）
        """
        self.device = device
        self.policy = policy.to(self.device)
        # policy_old 用于保存收集数据时的旧策略
        self.policy_old = type(policy)(act_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.act_dim = act_dim

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.clip_ratio = clip_ratio
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.gamma = gamma
        self.lam = lam
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

        # on-policy buffer (trajectory buffer). 每次 update 前会收集若干 step 的数据
        self.reset_buffer()

    def get_current_sigma(self):
        """
        返回当前策略（policy）估计的动作标准差的标量近似值，供日志使用。
        由于 policy.log_std_layer 是一个 Linear 层，我们使用其 bias 的平均作为近似。
        这个方法不会影响训练，仅用于监控。
        """
        with torch.no_grad():
            layer = getattr(self.policy, 'log_std_layer', None)
            if layer is None:
                return 1.0
            # 计算 softplus(s) 真实 sigma 约计
            if hasattr(layer, 'bias') and layer.bias is not None:
                return float(F.softplus(layer.bias.data).mean().cpu().item())
            if hasattr(layer, 'weight'):
                return float(F.softplus(layer.weight.data).mean().cpu().item())
            return 1.0


    def reset_buffer(self):
        """清空 on-policy buffer（必须在 episode 开始或 update 后调用以避免跨回合污染）"""
        self.buf_obs_img = []        # list of np arrays or tensors (H,W) or (1,H,W)
        self.buf_obs_state = []
        self.buf_actions = []
        self.buf_logp = []           # old log probs (来自 policy_old) — 存为标量张量
        self.buf_rewards = []
        self.buf_vals = []           # value estimates from policy_old at time of collection
        self.buf_dones = []
        self.path_start_idx = 0
        # Also clear computed returns / advantages if present to keep invariants:
        # After reset, buf_ret / buf_adv must match buf_rewards length (both zero).
        if hasattr(self, 'buf_adv'):
            self.buf_adv = []
        if hasattr(self, 'buf_ret'):
            self.buf_ret = []

    # ---------- utilities for tanh-squashed gaussian ----------
    @staticmethod
    def atanh(x):
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))

    def gaussian_logprob_raw(self, mu, std, raw_action):
        """计算 Normal(mu,std) 对 raw_action 的 log_prob（不含 tanh Jacobian）"""
        var = std.pow(2)
        log_scale = std.log()
        return -0.5 * (((raw_action - mu) ** 2) / var + 2 * log_scale + math.log(2 * math.pi))

    def log_prob_tanh_action(self, mu, std, action):
        """
        Safe tanh-squash log-prob implementation.
        Includes:
        - nan/inf sanitization
        - stable atanh
        - stable log-Jacobian
        - clamped std
        """

        # ---------------------------------------
        # 1. Sanitize inputs (avoid NaN propagation)
        # ---------------------------------------
        mu = torch.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
        std = torch.nan_to_num(std, nan=1e-3, posinf=1e6, neginf=1e-6)
        action = torch.nan_to_num(action, nan=0.0, posinf=0.999999, neginf=-0.999999)
        
        # clamp std for stability
        std = torch.clamp(std, 1e-6, 1e3)

        # ---------------------------------------
        # 2. Atanh (invert tanh) in a numerically stable way
        # ---------------------------------------
        # avoid atanh exploding at ±1:
        eps = 1e-6
        clipped = action.clamp(-1 + eps, 1 - eps)

        # correct definition of atanh:
        # atanh(x) = 0.5 * log((1+x)/(1-x))
        raw = 0.5 * (torch.log1p(clipped) - torch.log1p(-clipped))

        # ---------------------------------------
        # 3. Normal distribution for raw
        # ---------------------------------------
        try:
            normal = torch.distributions.Normal(mu, std)
        except Exception as e:
            print("\n[ERROR] Normal() creation failed in log_prob_tanh_action")
            print("  mu min/mean/max:", mu.min().item(), mu.mean().item(), mu.max().item())
            print("  std min/mean/max:", std.min().item(), std.mean().item(), std.max().item())
            raise

        logp_raw = normal.log_prob(raw)  # shape: [B, act_dim]

        # ---------------------------------------
        # 4. Tanh squash adjustment:
        #    logp = logp_raw - log(1 - tanh(raw)^2)
        # ---------------------------------------
        # ensure Jacobian is safe: 1 - clipped^2 ∈ (0,1)
        jacobian = 1 - clipped.pow(2) + 1e-6
        logp = logp_raw - torch.log(jacobian)

        # ---------------------------------------
        # 5. Final logp: sum over action dim
        # ---------------------------------------
        logp = logp.sum(dim=-1, keepdim=True)

        # ---------------------------------------
        # 6. Last safety layer (avoid NaN going out)
        # ---------------------------------------
        logp = torch.nan_to_num(logp, nan=0.0, posinf=-50.0, neginf=-50.0)

        return logp



    # ---------- data collection API ----------
    def choose_action(self, obs_img, obs_state, deterministic=False):
        """Convert input -> run policy_old -> sample/logp/value -> return numpy scalars."""
        # convert inputs to tensors
        if isinstance(obs_img, np.ndarray):
            img_t = torch.from_numpy(obs_img)
        else:
            img_t = obs_img

        if isinstance(obs_state, np.ndarray):
            st_t = torch.from_numpy(obs_state)
        elif isinstance(obs_state, list):
            st_t = torch.tensor(obs_state, dtype=torch.float32)
        else:
            st_t = obs_state

        # ensure dims
        if img_t.dim() == 2:
            img_t = img_t.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        elif img_t.dim() == 3:
            img_t = img_t.unsqueeze(0)               # (1,C,H,W)
        if st_t.dim() == 1:
            st_t = st_t.unsqueeze(0)

        img_t = img_t.float().to(self.device)
        st_t = st_t.float().to(self.device)

        with torch.no_grad():
            mu, sigma, value = self.policy_old(img_t, st_t)

            # sanitize outputs
            mu = torch.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
            sigma = torch.nan_to_num(sigma, nan=1e-3, posinf=1e6, neginf=1e-6)
            sigma = torch.clamp(sigma, min=1e-6, max=1e6)

            if deterministic:
                raw = mu
            else:
                dist = torch.distributions.Normal(mu, sigma)
                raw = dist.rsample()

            action_raw = torch.tanh(raw)  # in [-1, 1]
            action = 0.5 * action_raw      # scale to [-0.5, 0.5] for env constraints
            logp = self.log_prob_tanh_action(mu, sigma, action_raw)

            action_np = action.cpu().numpy().squeeze()
            logp_f = float(logp.cpu().numpy().squeeze())
            value_f = float(value.cpu().numpy().squeeze())

        return action_np, logp_f, value_f



    def store_transition_catch(self, obs_img, obs_state, action, logp, reward, value, done):
        """
        存储一步交互数据到 buffer。
        """

        # ---------- obs_img ----------
        if isinstance(obs_img, np.ndarray):
            img_t = torch.from_numpy(obs_img).float()
        elif isinstance(obs_img, torch.Tensor):
            img_t = obs_img.detach().cpu().float()
        else:
            raise TypeError(f"obs_img type invalid: {type(obs_img)}")

        # ---------- obs_state ----------
        if isinstance(obs_state, list):
            st_t = torch.tensor(obs_state, dtype=torch.float32)
        elif isinstance(obs_state, np.ndarray):
            st_t = torch.from_numpy(obs_state).float()
        elif isinstance(obs_state, torch.Tensor):
            st_t = obs_state.detach().cpu().float()
        else:
            raise TypeError(f"obs_state type invalid: {type(obs_state)}")

        # ---------- action ----------
        if isinstance(action, np.ndarray):
            a_t = torch.from_numpy(action).float()
        elif isinstance(action, torch.Tensor):
            a_t = action.detach().cpu().float()
        elif isinstance(action, (float, int, np.floating, np.integer)):
            a_t = torch.tensor([float(action)], dtype=torch.float32)
        else:
            raise TypeError(f"action type invalid: {type(action)}")

        # normalize action shape to (act_dim,) or scalar (1)
        if a_t.ndim == 0:
            a_t = a_t.unsqueeze(0)

        # ---------- store ----------
        self.buf_obs_img.append(img_t.clone())
        self.buf_obs_state.append(st_t.clone())
        self.buf_actions.append(a_t.clone())

        self.buf_logp.append(float(logp))
        self.buf_rewards.append(float(reward))
        self.buf_vals.append(float(value))
        self.buf_dones.append(bool(done))


    # ---------- finish path and GAE ----------
    def finish_path(self, last_value=0.0):
        """
        当一个 episode 结束或在中间截断要计算 returns/advantages 时调用。
        通过对当前 buffer 中从 path_start_idx 开始到末尾的段计算 GAE，并把 computed returns & advantages 存回 buffer。
        last_value: 如果不是终止（done==False），由当前策略估计的 next value；否则为 0。
        """
        path_slice = slice(self.path_start_idx, len(self.buf_rewards))
        rewards = np.array(self.buf_rewards[path_slice], dtype=np.float32)
        values = np.array(self.buf_vals[path_slice], dtype=np.float32)
        dones = np.array(self.buf_dones[path_slice], dtype=np.bool_)

        # append last_value to values for delta computation
        values_extended = np.append(values, last_value)
        
        # GAE
        advantages = np.zeros_like(rewards)
        lastgaelam = 0
        for t in reversed(range(len(rewards))):
            nonterminal = 1.0 - float(dones[t])
            delta = rewards[t] + self.gamma * values_extended[t + 1] * nonterminal - values_extended[t]
            lastgaelam = delta + self.gamma * self.lam * nonterminal * lastgaelam
            advantages[t] = lastgaelam

        returns = advantages + values

        # store as lists aligned with buffer
        # we will attach advantage & returns arrays to lists parallel to buf_rewards
        if not hasattr(self, 'buf_adv'):
            self.buf_adv = []
            self.buf_ret = []
        # extend with computed ones
        self.buf_adv.extend(advantages.tolist())
        self.buf_ret.extend(returns.tolist())

        # move path start
        self.path_start_idx = len(self.buf_rewards)
            
    # ---------- main update (PPO) ----------
    def learn(self):
        """
        对当前收集到的 on-policy 数据执行 PPO 更新。
        要求在调用前已对所有正在进行的 episode 调用 finish_path(last_value)（对未终止 episode 提供 bootstrap value）。
        """
        # collect all data as tensors
        n = len(self.buf_rewards)
        if n == 0:
            return dict(loss=0.0, policy_loss=0.0, value_loss=0.0, entropy=0.0)

        # ensure advantages and returns exist for all steps
        assert hasattr(self, 'buf_adv') and len(self.buf_adv) == n, "Must call finish_path for all trajectories before update()"

        # stack tensors and move to device
        obs_imgs = torch.stack([t.float() for t in self.buf_obs_img]).to(self.device)
        obs_states = torch.stack([t.float() for t in self.buf_obs_state]).to(self.device)
        actions = torch.stack([t.float() for t in self.buf_actions]).to(self.device)
        old_logp = torch.tensor(self.buf_logp, dtype=torch.float32, device=self.device)
        returns = torch.tensor(self.buf_ret, dtype=torch.float32, device=self.device)
        raw_advantages = torch.tensor(self.buf_adv, dtype=torch.float32, device=self.device)

        # 对 actor 保留一部分 advantage 的原始尺度信息。
        # 纯标准化会把不同 update 之间的策略改进强度压到近似同一量级，
        # 导致 policy_loss 长期围绕极小值波动，不利于观察真实变化。
        adv_mean = raw_advantages.mean()
        adv_std = raw_advantages.std(unbiased=False)
        advantages = (raw_advantages - adv_mean) / (adv_std + 1e-8)
        adv_scale = torch.clamp(adv_std.detach(), min=0.5, max=10.0)
        actor_advantages = advantages * adv_scale

        # --- Enhanced safety checks: skip update if returns/advantages contain NaN/Inf or are extreme ---
        try:
            if (not torch.isfinite(returns).all()) or (not torch.isfinite(raw_advantages).all()) or (not torch.isfinite(actor_advantages).all()):
                print("[PPO.learn] Skipping update: returns/advantages contain NaN/Inf")
                return dict(loss=0.0, policy_loss=0.0, value_loss=0.0, entropy=0.0, log_std_mean=0.0, log_std_grad_norm=0.0, mean_sigma=0.0, skipped=True)
            # 放宽阈值，从1e4提高到1e5，允许更多正常更新
            if returns.abs().max() > 1e5 or raw_advantages.abs().max() > 1e5 or actor_advantages.abs().max() > 1e5:
                print(
                    f"[PPO.learn] Skipping update: returns/advantages too large "
                    f"(returns_max={returns.abs().max().item():.2f}, raw_adv_max={raw_advantages.abs().max().item():.2f}, "
                    f"actor_adv_max={actor_advantages.abs().max().item():.2f})"
                )
                return dict(loss=0.0, policy_loss=0.0, value_loss=0.0, entropy=0.0, log_std_mean=0.0, log_std_grad_norm=0.0, mean_sigma=0.0, skipped=True)
        except Exception as e:
            # If any unexpected error during safety checks, skip update
            print(f"[PPO.learn] Safety check error: {e}, skipping update.")
            return dict(loss=0.0, policy_loss=0.0, value_loss=0.0, entropy=0.0, log_std_mean=0.0, log_std_grad_norm=0.0, mean_sigma=0.0, skipped=True)

        dataset = TensorDataset(obs_imgs, obs_states, actions, old_logp, returns, actor_advantages)
        dataloader = DataLoader(dataset, batch_size=self.minibatch_size, shuffle=True)

        # --- [新增] 在开始更新前保存 policy 备份，用于异常时回滚 ---
        from copy import deepcopy
        policy_backup = deepcopy(self.policy.state_dict())

        # update policy_old to current policy params BEFORE updating? No: PPO uses old policy stored at data collection time.
        # policy_old was saved when collecting data (we must ensure that before collecting we called policy_old.load_state_dict(policy.state_dict()))
        # We'll perform K epochs of SGD on the collected dataset
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        iters = 0

        for epoch in range(self.update_epochs):
            for batch in dataloader:
                b_imgs, b_states, b_actions, b_old_logp, b_returns, b_adv = [x.to(self.device) for x in batch]

                # 保证 logp 和 advantage 在计算比值时拥有匹配的形状 [B, 1]，
                # 否则广播会把每个样本同批次其它样本混在一起，导致 policy_loss 恒为 0。
                if b_old_logp.dim() == 1:
                    b_old_logp = b_old_logp.unsqueeze(-1)
                if b_adv.dim() == 1:
                    b_adv = b_adv.unsqueeze(-1)

                # new policy evaluation
                mu, sigma, value_pred = self.policy(b_imgs, b_states)  # shapes: [B,act_dim], [B,act_dim], [B,1]
                sigma = torch.clamp(sigma, 1e-6, 1e6)
                # compute new log prob for given (tanh-ed) actions
                # ---------- debug & defensive checks before computing logp ----------
                # b_imgs, b_states, b_actions, mu, sigma are tensors in shapes you expect
                # Print shapes and some stats if NaN or Inf detected
                if torch.isnan(mu).any() or torch.isinf(mu).any() or torch.isnan(sigma).any() or torch.isinf(sigma).any():
                    print("[PPO::learn] Detected NaN/Inf in policy outputs BEFORE computing logp.")
                    print(" mu stats: nan_any=%s, inf_any=%s, min/mean/max: %s/%s/%s" % (
                        torch.isnan(mu).any().item(), torch.isinf(mu).any().item(),
                        None if mu.numel()==0 else mu.min().item(), None if mu.numel()==0 else mu.mean().item(), None if mu.numel()==0 else mu.max().item()
                    ))
                    print(" sigma stats: nan_any=%s, inf_any=%s, min/mean/max: %s/%s/%s" % (
                        torch.isnan(sigma).any().item(), torch.isinf(sigma).any().item(),
                        None if sigma.numel()==0 else sigma.min().item(), None if sigma.numel()==0 else sigma.mean().item(), None if sigma.numel()==0 else sigma.max().item()
                    ))
                    # Also inspect inputs
                    print(" b_imgs shape:", getattr(b_imgs, "shape", None))
                    print(" b_states shape:", getattr(b_states, "shape", None))
                    print(" b_actions shape:", getattr(b_actions, "shape", None))
                    # Dump first few entries for mu/sigma
                    print(" mu[0:5]:", mu.view(-1)[:5].detach().cpu().numpy())
                    print(" sigma[0:5]:", sigma.view(-1)[:5].detach().cpu().numpy())
                    # Fallback: sanitize mu/sigma to safe values to allow training to continue
                    mu = torch.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
                    sigma = torch.nan_to_num(sigma, nan=1e-3, posinf=1e6, neginf=1e-6)
                    sigma = torch.clamp(sigma, 1e-6, 1e3)

                # 由于 choose_action 里用了 0.5 缩放，计算 logp 时需要还原到 tanh(raw) 的尺度
                new_logp = self.log_prob_tanh_action(mu, sigma, b_actions * 2.0)
                # ratio
                ratio = torch.exp(new_logp - b_old_logp)
                # clipped surrogate
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * b_adv
                policy_loss = -torch.mean(torch.min(surr1, surr2))
                # entropy (for exploration)
                normal = torch.distributions.Normal(mu, sigma)
                # differential entropy of normal: sum(log(sigma) + 0.5*log(2*pi*e))
                ent = normal.entropy().sum(dim=-1).mean()

                # value loss (MSE)
                value_pred = value_pred.view(-1)
                value_loss = F.mse_loss(value_pred, b_returns)

                loss = policy_loss - self.entropy_coef * ent + self.value_coef * value_loss

                # --- [修复] 确保每个 batch 后立即 step 和 zero_grad，防止梯度累积 ---
                self.optimizer.zero_grad()
                loss.backward()
                # gradient clipping
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # --- [增强] 每个 batch 后检查是否有异常，如果有则回滚并跳过整个更新 ---
                try:
                    # 检查是否有 NaN/Inf 在 loss 中，放宽阈值
                    if not torch.isfinite(loss) or loss.abs().item() > 1e4:
                        print(f"[PPO::learn] Detected invalid loss after step: {loss.item():.4f}. Rolling back parameters.")
                        self.policy.load_state_dict(policy_backup)
                        self.policy_old.load_state_dict(policy_backup)
                        return dict(loss=0.0, policy_loss=0.0, value_loss=0.0, entropy=0.0,
                                    log_std_mean=0.0, log_std_grad_norm=0.0, mean_sigma=0.0, skipped=True)

                    # 检查参数是否变得无效，放宽阈值
                    for name, param in self.policy.named_parameters():
                        if not torch.isfinite(param).all() or param.abs().max().item() > 1e5:
                            print(f"[PPO::learn] Detected invalid parameter '{name}' max={param.abs().max().item():.4f}. Rolling back parameters.")
                            self.policy.load_state_dict(policy_backup)
                            self.policy_old.load_state_dict(policy_backup)
                            return dict(loss=0.0, policy_loss=0.0, value_loss=0.0, entropy=0.0,
                                        log_std_mean=0.0, log_std_grad_norm=0.0, mean_sigma=0.0, skipped=True)
                except Exception as e:
                    print(f"[PPO::learn] Error during safety check: {e}. Rolling back parameters.")
                    self.policy.load_state_dict(policy_backup)
                    self.policy_old.load_state_dict(policy_backup)
                    return dict(loss=0.0, policy_loss=0.0, value_loss=0.0, entropy=0.0,
                                log_std_mean=0.0, log_std_grad_norm=0.0, mean_sigma=0.0, skipped=True)

                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += ent.item()
                iters += 1

        # after update, sync policy_old <- policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # 记录 log_std 的均值和梯度范数（如果存在）
        log_std_mean = 0.0
        log_std_grad_norm = 0.0
        mean_sigma = 0.0
        try:
            if hasattr(self.policy, 'log_std_layer'):
                with torch.no_grad():
                    bias = self.policy.log_std_layer.bias
                    log_std_mean = float(bias.mean().cpu().item())
                    mean_sigma = float(torch.exp(bias.mean()).cpu().item())
                # 梯度范数（可能为 None）
                grads = [p.grad.norm().cpu().item() for p in self.policy.log_std_layer.parameters() if p.grad is not None]
                if grads:
                    log_std_grad_norm = float(sum(grads))
        except Exception:
            pass

        # clear buffer
        self.reset_buffer()
        # also drop buf_adv/ret
        if hasattr(self, 'buf_adv'):
            try:
                del self.buf_adv
                del self.buf_ret
            except Exception:
                pass

        return dict(loss=total_loss / max(1, iters),
                    policy_loss=total_policy_loss / max(1, iters),
                    value_loss=total_value_loss / max(1, iters),
                    entropy=total_entropy / max(1, iters),
                    log_std_mean=log_std_mean,
                    log_std_grad_norm=log_std_grad_norm,
                    mean_sigma=mean_sigma)

    # ---------- helper to bootstrap unfinished episode ----------
    def finish_path_with_value(self, last_img, last_state):
        """
        当你在 update 前希望对仍未结束的 episode 做 bootstrap（即不是 done），
        使用当前 policy 计算 last_value 并调用 finish_path(last_value)
        """
        # prepare tensors
        if isinstance(last_img, np.ndarray):
            img_t = torch.tensor(last_img, dtype=torch.float32, device=self.device).unsqueeze(0)
        else:
            img_t = last_img.to(self.device).unsqueeze(0).float()

        if isinstance(last_state, np.ndarray):
            st_t = torch.tensor(last_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        else:
            st_t = last_state.to(self.device).unsqueeze(0).float()

        with torch.no_grad():
            _, _, last_value = self.policy_old(img_t, st_t)
            last_value = float(last_value.cpu().numpy().squeeze())
        self.finish_path(last_value=last_value)

    # ---------- convenience: call at start of data-collection to freeze old policy ----------
    def start_collection(self):
        """
        在开始新一轮数据收集前调用（把 policy 的当前参数 copy 到 policy_old）。
        典型流程：
          ppo.start_collection()
          for t in range(T): interact and store_transition(...)
          for each episode: finish_path(...)  # includes bootstrap for last incomplete traj
          ppo.update()
        """
        self.policy_old.load_state_dict(self.policy.state_dict())

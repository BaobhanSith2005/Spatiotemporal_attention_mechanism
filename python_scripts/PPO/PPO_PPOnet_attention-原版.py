from collections import deque
import gc  # 移到文件顶部，避免重复导入
import random  # 用于ReplayBuffer的sample方法

import torch
import torch.nn as nn
import numpy as np
import torch_geometric
from torch_geometric.data import Data
import torch.nn.functional as F
from python_scripts.Project_config import device
from Spatiotemporal_attention_mechanism.spatial_attention.CBAM import CBAM
from Spatiotemporal_attention_mechanism.spatial_attention.bidirectional_cross_attention import BidirectionalCrossAttention
from Spatiotemporal_attention_mechanism.temporal_attention.multihead_self_attention import MultiHeadSelfAttention

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


class ActorCritic(nn.Module):
    def __init__(self, act_dim):
        super().__init__()
        self.device = device  # 设置设备属性
        self.buffer_img = deque(maxlen=20)
        self.buffer_state = deque(maxlen=20)

        self.relu = nn.ReLU()
        # 图像处理部分 - 优化配置减少内存占用
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  # 128→64
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # 64→32 - 大幅减少通道数
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 32 * 32, 512)  # 更新维度并减少神经元数量
        self.dropout = nn.Dropout(0.2)  # 减少dropout率以保留更多信息
        self.fc2 = nn.Linear(512, 256)  # 降维到256
        self.fc3 = nn.Linear(256, 128)  # 进一步降维
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # 舵机角度处理部分
        self.conv_graph1 = torch_geometric.nn.SAGEConv(1, 1024 // 4, normalize=True)
        self.bn_graph1 = torch_geometric.nn.BatchNorm(1024 // 4)
        self.conv_graph2 = torch_geometric.nn.GATConv(1024 // 4, 1024 // 2, heads=4)
        self.bn_graph2 = torch_geometric.nn.BatchNorm(4 * 1024 // 2)  # 多头输出维度变化
        self.conv_graph3 = torch_geometric.nn.SAGEConv(4 * 1024 // 2, 1024)
        self.bn_graph3 = torch_geometric.nn.BatchNorm(1024)
        self.conv_graph4 = torch_geometric.nn.GATConv(1024, 1024, heads=1)  # 单头输出
        self.bn_graph4 = torch_geometric.nn.BatchNorm(1024)
        self.conv_graph5 = torch_geometric.nn.GCNConv(1024, 512)
        self.fc4 = nn.Linear(512, 128)

        # 空间注意力部分，针对图像
        self.cbam1 = CBAM(16)
        self.cbam2 = CBAM(32)
        self.cbam3 = CBAM(64)
        self.cbam4 = CBAM(64)

        # 时间注意力部分
        self.tem_attn1 = MultiHeadSelfAttention(128, 4, 0.1)
        self.tem_attn2 = MultiHeadSelfAttention(128, 4, 0.1)

        # 交叉注意力融合部分
        self.spatial_vision_graph_attn = BidirectionalCrossAttention(  # 图像-舵机角度空间交互
            dim=128, heads=4, dim_head=32, prenorm=True
        )
        self.spatial_fusion_attn = BidirectionalCrossAttention(  # 空间融合
            dim=256, heads=8, dim_head=32, prenorm=True
        )
        self.temporal_vision_graph_attn = BidirectionalCrossAttention(  # 图像-舵机角度时间交互
            dim=128, heads=4, dim_head=32, prenorm=True
        )
        self.temporal_fusion_attn = BidirectionalCrossAttention(  # 时间融合
            dim=256, heads=8, dim_head=32, prenorm=True
        )
        self.final_fusion_attn_front = BidirectionalCrossAttention(  # 时空融合
            dim=256, heads=8, dim_head=32, prenorm=True
        )
        self.final_fusion_attn = BidirectionalCrossAttention(  # 时空融合
            dim=512, heads=16, dim_head=32, prenorm=True
        )

        # 决策层
        self.decision_layer = DecisionLayer(input_dim=512, hidden_dim=256, output_dim=1)
        self.critic = nn.Linear(512, 1)
        self.mu_layer = nn.Linear(512, act_dim)
        self.log_std_layer = nn.Linear(512, act_dim)
        # 添加参数限制范围，避免NaN值
        self.log_std_min = -20  # 最小对数标准差，exp(-20)≈2e-9
        self.log_std_max = 2    # 最大对数标准差，exp(2)≈7.39


    # 将图像特征存入缓冲区
    def update_buffer_img(self, feature):
        self.buffer_img.append(feature.detach())

    # 将舵机角度特征存入缓冲区
    def update_buffer_state(self, feature):
        self.buffer_state.append(feature.detach())

    # 时间注意力前处理
    def _process_temporal(self, buffer, new_feature):
        sequence = torch.stack([b.to(new_feature.device) for b in buffer], dim=1)  # [B, T, D]
        return sequence

    def forward(self, img, state):
        """
        参数:
        - img: 输入图像，形状为 [1, 128, 128]
        - state: 输入状态，形状为 [4]
        """
        # 提取图像初始特征
        y = self.conv1(img)
        y = self.cbam1(y)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.conv2(y)
        y = self.cbam2(y)
        y = self.bn2(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.conv3(y)
        y = self.cbam3(y)
        y = self.bn3(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.conv4(y)
        y = self.cbam4(y)
        y = self.bn4(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.flatten(y)
        y = self.fc1(y)
        y = self.fc2(y)
        y = self.fc3(y)
        
        # 归一化处理，支持批量和单个输入
        # 对于批量输入，对每个样本单独归一化
        if len(y.shape) > 1:
            batch_size = y.shape[0]
            normalized_features = []
            
            for i in range(batch_size):
                min_val = torch.min(y[i])
                max_val = torch.max(y[i])
                denominator = torch.sub(max_val, min_val) + 1e-8
                norm_y = torch.div(torch.sub(y[i], min_val), denominator)
                normalized_features.append(norm_y.unsqueeze(0))
            
            # 合并批量特征并添加通道维度
            initial_image_features = torch.cat(normalized_features, dim=0).unsqueeze(1)  # [batch_size, 1, 128]
        else:
            # 单个输入的处理
            min_val1 = torch.min(y)  # 最小值
            max_val1 = torch.max(y)  # 最大值
            denominator = torch.sub(max_val1, min_val1) + 1e-8  # 1e-8 是一个很小的数
            initial_image_features = torch.div(torch.sub(y, min_val1), denominator).unsqueeze(1)  # [1, 1, 128]


        # 提取舵机初始特征
        graph_result = self.creat_graph(state)
        
        # 处理批量或单个输入
        if isinstance(graph_result, list):  # 批量输入
            batch_size = len(graph_result)
            batch_features = []
            
            # 对每个样本单独处理
            for graph in graph_result:
                x = graph.x
                edge_index = graph.edge_index
                
                # 应用图卷积层
                x = self.conv_graph1(x, edge_index)
                x = self.bn_graph1(x)
                x = self.relu(x)
                x = self.dropout(x)
                x = self.conv_graph2(x, edge_index)
                x = self.bn_graph2(x)
                x = self.relu(x)
                x = self.dropout(x)
                x = self.conv_graph3(x, edge_index)
                x = self.bn_graph3(x)
                x = self.relu(x)
                x = self.dropout(x)
                x = self.conv_graph4(x, edge_index)
                x = self.bn_graph4(x)
                x = self.relu(x)
                x = self.dropout(x)
                x = self.conv_graph5(x, edge_index)
                x = self.fc4(x)
                
                # 归一化处理
                min_val = torch.min(x)
                max_val = torch.max(x)
                denominator = torch.sub(max_val, min_val) + 1e-8
                x_norm = torch.div(torch.sub(x, min_val), denominator)
                
                batch_features.append(x_norm.unsqueeze(0))  # 添加批次维度
            
            # 合并所有批次的特征
            initial_state_features = torch.cat(batch_features, dim=0)  # [batch_size, 4, 128]
        else:  # 单个输入
            graph = graph_result
            x_graph = graph.x
            edge_index = graph.edge_index
            
            x_graph = self.conv_graph1(x_graph, edge_index)
            x_graph = self.bn_graph1(x_graph)
            x_graph = self.relu(x_graph)
            x_graph = self.dropout(x_graph)
            x_graph = self.conv_graph2(x_graph, edge_index)
            x_graph = self.bn_graph2(x_graph)
            x_graph = self.relu(x_graph)
            x_graph = self.dropout(x_graph)
            x_graph = self.conv_graph3(x_graph, edge_index)
            x_graph = self.bn_graph3(x_graph)
            x_graph = self.relu(x_graph)
            x_graph = self.dropout(x_graph)
            x_graph = self.conv_graph4(x_graph, edge_index)
            x_graph = self.bn_graph4(x_graph)
            x_graph = self.relu(x_graph)
            x_graph = self.dropout(x_graph)
            x_graph = self.conv_graph5(x_graph, edge_index)
            x_graph = self.fc4(x_graph)
            min_val2 = torch.min(x_graph)  # 最小值
            max_val2 = torch.max(x_graph)  # 最大值
            denominator2 = torch.sub(max_val2, min_val2) + 1e-8  # 1e-8 是一个很小的数
            initial_state_features = torch.div(torch.sub(x_graph, min_val2), denominator2).unsqueeze(0) # [1, 4, 128]
        
        # 空间注意力融合 - 支持批量处理
        vision_graph_out, graph_vision_out = self.spatial_vision_graph_attn(initial_image_features, initial_state_features)
        
        # 对于批量输入，需要为每个样本重复相应维度
        batch_size = initial_image_features.shape[0]
        vision_graph_out = vision_graph_out.repeat(1, graph_vision_out.shape[1], 1)
        spatial_features = torch.cat([vision_graph_out, graph_vision_out], dim=-1)
        spatial_features_out, _ = self.spatial_fusion_attn(spatial_features, spatial_features) # [batch_size, 4, 256]

        # 时间注意力融合 - 更新缓冲区，避免计算图累积
        # 对于批量输入，我们只使用第一个样本更新缓冲区（或者根据需要调整）
        if batch_size > 0:
            self.update_buffer_img(initial_image_features[0:1])  # 只使用第一个样本
            self.update_buffer_state(initial_state_features[0:1])  # 只使用第一个样本
        
        # 时间特征处理 - 支持批量和单个输入
        if len(self.buffer_img) > 0:
            # 对于形状为[1,1,128]的特征，应该沿dim=0连接以形成时间序列
            sequence_img = torch.cat(list(self.buffer_img), dim=0)  # 连接后形状: [N, 1, 128]，其中N是队列长度
            # 调整维度顺序以匹配注意力机制的输入要求
            temporal_attn_out = self.tem_attn1(sequence_img.permute(1, 0, 2)).permute(1, 0, 2)
            # 取最后一个时间步
            temporal_img_out = temporal_attn_out[-1:, :, :]  # [1,1,128]
            
            # 对于批量输入，复制到所有样本
            if batch_size > 1:
                temporal_img_out = temporal_img_out.repeat(batch_size, 1, 1)  # [batch_size, 1, 128]
        else:
            # 安全处理：如果队列为空，使用当前特征
            temporal_img_out = initial_image_features

        if len(self.buffer_state) > 0:
            # 对于形状为[1,4,128]的特征，应该沿dim=0连接以形成时间序列
            sequence_state = torch.cat(list(self.buffer_state), dim=0)  # 连接后形状: [N, 4, 128]，其中N是队列长度
            # 调整维度顺序以匹配注意力机制的输入要求
            temporal_attn_out = self.tem_attn2(sequence_state.permute(1, 0, 2)).permute(1, 0, 2)
            # 取最后一个时间步
            temporal_state_out = temporal_attn_out[-1:, :, :]  # [1,4,128]
            
            # 对于批量输入，复制到所有样本
            if batch_size > 1:
                temporal_state_out = temporal_state_out.repeat(batch_size, 1, 1)  # [batch_size, 4, 128]
        else:
            # 安全处理：如果队列为空，使用当前特征
            temporal_state_out = initial_state_features

        # 时间域的视觉-图融合
        temporal_img_out, temporal_state_out = self.temporal_vision_graph_attn(temporal_img_out, temporal_state_out)
        temporal_img_out = temporal_img_out.repeat(1, temporal_state_out.shape[1], 1)
        temporal_features = torch.cat([temporal_img_out, temporal_state_out], dim=-1)
        temporal_features_out, _ = self.temporal_fusion_attn(temporal_features, temporal_features) # [batch_size, 4, 256]
        
        # 时空注意力融合
        final_img_out, final_state_out = self.final_fusion_attn_front(temporal_features_out, spatial_features_out) # [batch_size, 4, 256]
        final_features = torch.cat([final_img_out, final_state_out], dim=-1)
        final_features_out, _ = self.final_fusion_attn(final_features, final_features) # [batch_size, 4, 512]
        
        # 池化处理 - 支持批量
        pooled_features = final_features_out.mean(dim=1)  # [batch_size, 512]
        
        # 输出层 - 支持批量
        mu = self.mu_layer(pooled_features)  # [batch_size, action_dim]
        log_std = self.log_std_layer(pooled_features)  # [batch_size, action_dim]
        # 裁剪log_std值，确保数值稳定性，防止exp操作产生NaN
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        sigma = torch.exp(log_std)
        value = self.critic(pooled_features)  # [batch_size, 1]
        
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
    def __init__(self,env_information):
        # 初始化环境信息
        self.act_dim_shoulder = env_information["action_dim_shoulder"]
        self.act_dim_arm = env_information["action_dim_arm"]
        
        # 创建策略网络和价值网络
        self.policy = ActorCritic(self.act_dim_shoulder + self.act_dim_arm).to(device)
        self.policy_old = ActorCritic(self.act_dim_shoulder + self.act_dim_arm).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # 设置优化器和学习率
        self.lr = 2e-4
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        
        # 学习率调度器
        self.lr_decay = 0.995  # 学习率衰减
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.lr_decay)
        
        # 初始化经验池，使用deque的maxlen参数自动管理缓冲区大小
        self.buffer = ReplayBuffer(capacity=50)  # 设置为较小的容量以匹配之前的需求
        
        # 设置PPO超参数
        self.clip_param = 0.2  # 裁剪参数
        self.max_grad_norm = 1.0  # 梯度裁剪阈值
        self.lam = 0.95  # GAE参数
        self.gamma = 0.99  # 折扣因子
        self.entropy_coef = 0.01  # 熵正则化系数
        self.value_coef = 0.5  # 价值函数损失系数
        self.gae_lambda = 0.95  # GAE参数
        self.clip_ratio = 0.2  # PPO裁剪参数
        self.update_epochs = 10  # 每批数据的更新次数
        self.batch_size = 64  # 批大小

    def clear_memory(self):
        """清空所有存储的轨迹数据，释放内存"""
        self.buffer.clear()

    def choose_action(self, episode_num, obs, explore=None):
        # 将输入转换为张量并移至正确设备
        # 添加批次维度以匹配网络期望的4D输入 [batch_size, channels, height, width]
        # 将输入转换为张量并移至正确设备
        # 使用clone().detach()以避免PyTorch警告
        if isinstance(obs[0], torch.Tensor):
            img = obs[0].clone().detach().to(device, dtype=torch.float32).unsqueeze(0)
        else:
            img = torch.tensor(obs[0], dtype=torch.float32, device=device).unsqueeze(0)
            
        if isinstance(obs[1], torch.Tensor):
            state = obs[1].clone().detach().to(device, dtype=torch.float32)
        else:
            state = torch.tensor(obs[1], dtype=torch.float32, device=device)

        # 使用递减的epsilon，从0.9开始逐渐降低到0.1
        epsilon = max(0.1, 0.90 - episode_num * 0.0001)

        # 如果显式指定了explore参数，则使用该值
        if explore is not None:
            use_random = explore
        else:
            # 否则根据epsilon决定是否探索
            random_num = np.random.uniform()
            use_random = random_num < epsilon
        
        with torch.no_grad():
            # 策略网络输出动作分布的均值 mu 和价值 value
            mu , sigma , value = self.policy(img, state)
            # 创建一个正态分布，用于计算对数概率和在利用时采样
            dist_shoulder = torch.distributions.Normal(mu, sigma)

            if use_random:
                # 探索：在[-1, 1]范围内随机选择动作
                act_dim = 1
                action_scaled = torch.tensor(np.random.uniform(-1, 1, size=act_dim), dtype=torch.float32).to(device)
            else:
                # 利用：从策略网络生成的分布中采样
                action_raw = dist_shoulder.sample()
                action_scaled = torch.tanh(action_raw)

                # 无论动作如何选择，都计算其在当前策略下的对数概率
                # 这是 on-policy 算法的要求
            action_raw_for_log_prob = torch.atanh(torch.clamp(action_scaled, -0.9999, 0.9999))
            log_prob = dist_shoulder.log_prob(action_raw_for_log_prob).sum(axis=-1)

            # 返回选择的动作、其对数概率和状态值
            return action_scaled.cpu().numpy(), log_prob.item(), value.item()


    def store_transition_catch(self, state, action_shoulder, action_arm, reward, next_state, done, value, log_prob_shoulder, log_prob_arm):
        """
        存储抓取阶段的经验，使用ReplayBuffer管理内存
        """
        # 确保所有数据都不保留计算图
        if isinstance(action_shoulder, torch.Tensor):
            action_shoulder = action_shoulder.detach().cpu().numpy()
        if isinstance(action_arm, torch.Tensor):
            action_arm = action_arm.detach().cpu().numpy()
        if isinstance(reward, torch.Tensor):
            reward = reward.detach().cpu().numpy()
        if isinstance(done, torch.Tensor):
            done = done.detach().cpu().numpy()
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        if isinstance(log_prob_shoulder, torch.Tensor):
            log_prob_shoulder = log_prob_shoulder.detach().cpu().numpy()
        if isinstance(log_prob_arm, torch.Tensor):
            log_prob_arm = log_prob_arm.detach().cpu().numpy()
        
        # 创建转换字典
        transition_dict = {
            'state': state,
            'action_shoulder': action_shoulder,
            'action_arm': action_arm,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'value': value,
            'log_prob_shoulder': log_prob_shoulder,
            'log_prob_arm': log_prob_arm
        }
        
        # 存储到ReplayBuffer并记录
        self.buffer.push(transition_dict)

    def calculate_advantages(self):
        """
        从ReplayBuffer中提取数据并计算GAE优势函数
        """
        if len(self.buffer) == 0:
            print("警告: ReplayBuffer为空，无法计算优势函数")
            return np.array([]), np.array([])
        
        # 使用ReplayBuffer的sample方法获取所有数据
        buffer_data = self.buffer.sample(len(self.buffer))
        rewards = np.array([transition['reward'] for transition in buffer_data])
        values = np.array([transition['value'] for transition in buffer_data])
        dones = np.array([transition['done'] for transition in buffer_data])
        
        # 检查NaN值
        if np.any(np.isnan(rewards)):
            print("NaN found in rewards")
        if np.any(np.isnan(values)):
            print("NaN found in values")
        if np.any(np.isnan(dones)):
            print("NaN found in dones")
        
        # 使用compute_gae方法计算优势函数
        advantages = self.compute_gae(rewards, values, values[-1] if len(values) > 0 and not dones[-1] else 0, dones)
        
        # 计算回报
        returns = advantages + values

        # 标准化优势函数 - 添加数组长度检查，避免空数组或单元素数组导致的计算错误
        if len(advantages) > 1:
            mean_adv = advantages.mean()
            std_adv = advantages.std()
            # 检查标准差是否为0
            if std_adv > 1e-8:
                advantages = (advantages - mean_adv) / (std_adv + 1e-8)
            else:
                advantages = advantages - mean_adv  # 如果标准差为0，只减去均值
        # 如果只有0或1个元素，则不进行标准化

        return advantages, returns
    
    def compute_gae(self, rewards, values, next_value, dones):
        """
        计算广义优势估计（GAE）
        
        :param rewards: 奖励数组
        :param values: 价值估计数组
        :param next_value: 下一个状态的价值估计
        :param dones: 完成标志数组
        :return: 优势估计数组
        """
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        
        # 从后向前计算优势函数
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                # 对于最后一个时间步，使用next_value
                delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            else:
                delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            
            advantages[t] = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]
        
        return advantages
    
    def process_transition_dict(self, transition_dict):
        """
        处理转换字典，将不同类型的数据统一转换为张量
        
        :param transition_dict: 包含状态、动作、奖励等信息的字典
        :return: 处理后的字典，所有值都转换为张量
        """
        processed_dict = {}

        for key in transition_dict:
            if isinstance(transition_dict[key], list):
                # 检查列表中的元素是否为PyTorch张量
                if transition_dict[key] and isinstance(transition_dict[key][0], torch.Tensor):
                    # 如果是张量列表，直接使用torch.cat合并它们
                    tensor = torch.cat(transition_dict[key])
                else:
                    # 否则转换为NumPy数组再转换为张量
                    np_array = np.array(transition_dict[key])
                    tensor = torch.tensor(np_array, dtype=torch.float)

                # 特殊处理 'states' 和 'next_states'
                if key == 'states' or key == 'next_states':
                    # 确保图像数据的形状一致
                    if len(tensor.shape) != 4:
                        raise ValueError(f"Unexpected shape for {key}: {tensor.shape}")
            elif isinstance(transition_dict[key], torch.Tensor):
                tensor = transition_dict[key].float()
            else:
                raise ValueError(f"Unsupported type for key '{key}': {type(transition_dict[key])}")

            processed_dict[key] = tensor
        return processed_dict

    def get_current_sigma(self):
        """
        仅用于日志记录的 sigma 估计，不改变网络结构。
        这里用 log_std_layer 的 bias 的平均值来近似当前 log_sigma。
        """
        with torch.no_grad():
            # log_std_layer: Linear(512 -> act_dim)，bias 形状是 [act_dim]
            log_std_bias = self.policy.log_std_layer.bias  # Tensor [act_dim]
            log_std_mean = log_std_bias.mean()
            return torch.exp(log_std_mean).item()
               
    def learn(self, action_type: str):
        """
        根据指定的动作类型（'shoulder' 或 'arm'）更新策略网络。

        :param action_type: 一个字符串，'shoulder' 或 'arm'，用于指定要更新哪个动作部分。
        :return: 平均损失值
        """
        # 检查ReplayBuffer是否有足够的数据
        buffer_size = len(self.buffer)
        if buffer_size < 4:  # 设置最小样本数
            print(f"经验不足，当前样本数: {buffer_size}，跳过学习")
            return 0, 0, 0, 0
        print(f"开始学习，采样...")
        # 从经验池中采样一批数据（批次大小为10），非破坏性采样
        batch_size = min(buffer_size, 10)
        # 确保sample方法不会修改原始缓冲区
        sample_transitions = self.buffer.sample(batch_size)
        
        # 初始化损失记录器
        total_loss = 0
        policy_losses = []
        value_losses = []
        entropy_values = []
        
        # 合并采样的数据
        merged_transition = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': [],
            'log_probs': [],
            'values': []
        }
        
        # 根据action_type选择相应的动作和对数概率键
        action_key = 'action_shoulder' if action_type == 'shoulder' else 'action_arm'
        log_prob_key = 'log_prob_shoulder' if action_type == 'shoulder' else 'log_prob_arm'
        
        for t in sample_transitions:
            # 处理状态数据 - 提取图像和状态特征
            img_data, state_data = t['state']
            
            # 确保数据形状正确
            if isinstance(img_data, np.ndarray):
                # 确保图像数据是4D形状 [batch_size, channels, height, width]
                if len(img_data.shape) == 2:  # (H, W) -> (1, 1, H, W)
                    img_data = img_data.reshape(1, 1, *img_data.shape)
                elif len(img_data.shape) == 3:  # 处理不同的3D格式
                    img_data = img_data.reshape(1, 1, *img_data.shape[-2:])
            
            # 确保状态数据长度为4
            if isinstance(state_data, (list, np.ndarray)) and len(state_data) > 4:
                state_data = state_data[:4]
            
            # 合并图像和状态数据作为完整状态
            full_state = (img_data, state_data)
            merged_transition['states'].append(full_state)
            
            # 添加其他数据
            merged_transition['actions'].append(t[action_key])
            merged_transition['rewards'].append(t['reward'])
            merged_transition['next_states'].append(t['next_state'])
            merged_transition['dones'].append(t['done'])
            merged_transition['log_probs'].append(t[log_prob_key])
            merged_transition['values'].append(t['value'])
        
        # 准备状态批次数据用于前向传播
        states_img = []
        states_feature = []
        for img, feature in merged_transition['states']:
            # 转换为张量
            if isinstance(img, np.ndarray):
                img_tensor = torch.from_numpy(img.astype(np.float32)).clone().to(device)
            else:
                img_tensor = torch.tensor(img, dtype=torch.float32).to(device)
            
            if isinstance(feature, (list, np.ndarray)):
                feature_tensor = torch.tensor(feature, dtype=torch.float32).to(device)
            else:
                feature_tensor = torch.tensor([feature], dtype=torch.float32).to(device)
            
            # 确保状态特征长度为4
            if len(feature_tensor) < 4:
                feature_tensor = torch.cat([feature_tensor, torch.zeros(4 - len(feature_tensor), device=device)])
            elif len(feature_tensor) > 4:
                feature_tensor = feature_tensor[:4]
            
            states_img.append(img_tensor)
            states_feature.append(feature_tensor)
        
        # 堆叠为批次
        states_img_batch = torch.cat(states_img)
        states_feature_batch = torch.stack(states_feature)
        
        # 处理动作和对数概率
        actions = torch.tensor([float(a.flatten()[0]) if isinstance(a, np.ndarray) else float(a) 
                                for a in merged_transition['actions']], dtype=torch.float32).to(device)
        old_log_probs = torch.tensor(merged_transition['log_probs'], dtype=torch.float32).to(device)
        
        # 计算GAE优势估计
        rewards = np.array(merged_transition['rewards'])
        values = np.array(merged_transition['values'])
        dones = np.array(merged_transition['dones'])
        
        # 计算最后一个状态的价值估计（用于GAE）
        last_img, last_feature = merged_transition['states'][-1]
        with torch.no_grad():
            if isinstance(last_img, np.ndarray):
                last_img_tensor = torch.from_numpy(last_img.astype(np.float32)).clone().to(device)
            else:
                last_img_tensor = torch.tensor(last_img, dtype=torch.float32).to(device)
            
            if isinstance(last_feature, (list, np.ndarray)):
                last_feature_tensor = torch.tensor(last_feature, dtype=torch.float32).to(device)
            else:
                last_feature_tensor = torch.tensor([last_feature], dtype=torch.float32).to(device)
            
            # 确保长度为4
            if len(last_feature_tensor) < 4:
                last_feature_tensor = torch.cat([last_feature_tensor, torch.zeros(4 - len(last_feature_tensor), device=device)])
            elif len(last_feature_tensor) > 4:
                last_feature_tensor = last_feature_tensor[:4]
            
            _, _, last_value = self.policy(last_img_tensor, last_feature_tensor)
            last_value_np = last_value.cpu().numpy()[0]
        
        # 使用compute_gae方法计算优势
        advantages = self.compute_gae(rewards, values, last_value_np, dones)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
        returns = advantages + torch.tensor(values, dtype=torch.float32).to(device)
        
        # 标准化优势函数
        if len(advantages) > 1:
            mean_adv = advantages.mean()
            std_adv = advantages.std()
            if std_adv > 1e-8:
                advantages = (advantages - mean_adv) / (std_adv + 1e-8)
            else:
                advantages = advantages - mean_adv
        
        # 更新策略网络和价值网络多次
        total_loss = 0
        policy_losses = []
        value_losses = []
        entropy_values = []
        
        update_epochs = min(self.update_epochs, 4)  # 最多4轮更新
        
        for epoch in range(update_epochs):
            # 前向传播获取当前策略的输出
            with torch.set_grad_enabled(True):
                # 获取当前策略的均值、标准差和价值估计
                # 批量处理所有样本，提高计算效率
                # 调整图像批次形状，确保为[batch_size, 1, 128, 128]格式
                batch_imgs = states_img_batch.clone()
                # 移除不必要的批次维度并确保正确的通道维度
                if batch_imgs.dim() == 5:  # 假设原始形状为[batch_size, 1, 1, 128, 128]
                    batch_imgs = batch_imgs.squeeze(1)
                elif batch_imgs.dim() == 4 and batch_imgs.shape[1] == 1:  # [batch_size, 1, 128, 128]
                    pass  # 已经是正确格式
                elif batch_imgs.dim() == 4 and batch_imgs.shape[1] > 1:  # 通道维度不在正确位置
                    batch_imgs = batch_imgs.permute(0, 3, 1, 2)  # 假设原始为[batch_size, 128, 128, 1]
                elif batch_imgs.dim() == 3:  # [batch_size, 128, 128]
                    batch_imgs = batch_imgs.unsqueeze(1)  # 添加通道维度
                
                # 批量调用policy处理所有样本
                means, stds, values_pred = self.policy(batch_imgs, states_feature_batch)
                
                # 检查是否有NaN值
                if torch.isnan(means).any() or torch.isnan(stds).any():
                    print(f"第{epoch+1}轮更新中检测到NaN值")
                    print(f"Means: {means}")
                    print(f"Stds: {stds}")
                    continue
                
                # 创建正态分布
                dist = torch.distributions.Normal(means, stds)
                
                # 调整actions的形状以匹配分布的batch_shape+event_shape
                # 确保actions和means的维度一致
                if len(means.shape) == 2:  # means形状为[batch_size, action_dim]
                    # 将actions从[batch_size]调整为[batch_size, action_dim]
                    actions_reshaped = actions.view(-1, 1).expand_as(means)
                else:  # 原始的三维情况
                    actions_reshaped = actions.view(-1, 1, 1).expand_as(means)
                
                # 计算新的对数概率
                log_probs = dist.log_prob(actions_reshaped).sum(dim=-1)
                
                # 计算PPO目标，添加安全措施防止NaN
                log_prob_diff = log_probs - old_log_probs
                # 裁剪log差值，防止指数操作溢出或产生NaN
                log_prob_diff = torch.clamp(log_prob_diff, -20, 20)
                ratios = torch.exp(log_prob_diff)
                # 进一步裁剪ratios值，确保稳定性
                ratios = torch.clamp(ratios, 0.1, 10.0)
                
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 添加熵正则化
                entropy = dist.entropy().mean()
                policy_loss -= self.entropy_coef * entropy
                
                # 计算价值损失
                value_loss = F.mse_loss(values_pred.squeeze(-1), returns)
                
                # 组装总损失
                loss = policy_loss + self.value_coef * value_loss
            
            # 更新策略网络
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward(retain_graph=False)
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 记录损失值
            loss_val = loss.item()
            policy_loss_val = policy_loss.item()
            value_loss_val = value_loss.item()
            entropy_val = entropy.item()
            
            total_loss += loss_val
            policy_losses.append(policy_loss_val)
            value_losses.append(value_loss_val)
            entropy_values.append(entropy_val)
            
            # print(f"第{epoch+1}/{update_epochs}轮更新 - 损失: {loss_val:.4f}, 策略损失: {policy_loss_val:.4f}, 价值损失: {value_loss_val:.4f}")
        
        # 计算平均损失值
        avg_loss = total_loss / len(policy_losses) if policy_losses else 0
        avg_policy_loss = sum(policy_losses) / len(policy_losses) if policy_losses else 0
        avg_value_loss = sum(value_losses) / len(value_losses) if value_losses else 0
        avg_entropy = sum(entropy_values) / len(entropy_values) if entropy_values else 0
        
        # 更新学习率（如果使用学习率调度器）
        if hasattr(self, 'scheduler') and self.scheduler is not None:
            self.scheduler.step()
        
        # 返回平均损失值
        return avg_loss, avg_policy_loss, avg_value_loss, avg_entropy
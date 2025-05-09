import torch
import torch.nn as nn
import numpy as np

class LearnableMaskTemporalAggregation(nn.Module):
    """
    实现可学习的时间聚合机制，通过上三角掩码矩阵自适应地聚合历史信息
    
    该模块能够在每个时间步自适应地强化重要关节并削弱冗余关节，
    通过学习一个上三角掩码矩阵来分配历史动作中不同关节的重要性。
    """
    def __init__(self, seq_len, joint_dim, init_weight_decay=0.8):
        """
        初始化可学习的时间聚合模块
        
        参数:
            seq_len: 序列长度 N  即输入序列长度
            joint_dim: 关节特征维度 J
            init_weight_decay: 预定义权重的衰减率，控制历史帧的初始权重分布
        """
        super(LearnableMaskTemporalAggregation, self).__init__()
        
        # 创建预定义的上三角权重矩阵 (N x N)
        # 确保每一行的权重和为1，且权重随时间距离增加而减小
        self.init_upper_triangular_weights(seq_len, init_weight_decay)
        
        # 创建可学习的偏移矩阵，为每个关节单独学习时间聚合权重
        # 形状为 [joint_dim, seq_len, seq_len]，初始为0
        self.learnable_offset = nn.Parameter(torch.zeros(joint_dim, seq_len, seq_len))
        
    def init_upper_triangular_weights(self, seq_len, decay_rate):
        """
        初始化上三角权重矩阵
        
        参数:
            seq_len: 序列长度
            decay_rate: 权重衰减率
        """
        # 创建上三角权重矩阵，确保每行权重和为1
        predefined_weights = torch.zeros(seq_len, seq_len)
        
        for i in range(seq_len):
            # 只考虑当前帧及之前的帧 (上三角矩阵)
            weights = torch.zeros(seq_len)
            
            # 为每一帧分配权重，越近的帧权重越大
            for j in range(i + 1):
                weights[j] = decay_rate ** (i - j)
            
            # 归一化权重，确保每行和为1
            if weights.sum() > 0:
                weights = weights / weights.sum()
            
            predefined_weights[i] = weights
        
        # 注册预定义权重为缓冲区（不参与梯度更新）
        self.register_buffer('predefined_weights', predefined_weights)
        
    def forward(self, x):
        """
        前向传播函数
        
        参数:
            x: 输入序列 [batch_size, seq_len, joint_dim]
            
        返回:
            聚合后的序列 [batch_size, seq_len, joint_dim]
        """
        batch_size, seq_len, joint_dim = x.shape
        
        # 确保序列长度与初始化时一致
        assert seq_len == self.predefined_weights.shape[0], f"输入序列长度 {seq_len} 与初始化长度 {self.predefined_weights.shape[0]} 不匹配"
        
        # 对每个关节单独计算并应用掩码
        aggregated_features = []
        
        for j in range(joint_dim):
            # 1. 获取当前关节的可学习偏移
            offset_j = self.learnable_offset[j]
            
            # 2. 计算最终掩码矩阵 = 预定义权重 + 可学习偏移
            # 使用softmax确保每行和为1，保持上三角特性
            mask_j = self.predefined_weights + offset_j
            mask_j = torch.softmax(mask_j, dim=1)
            
            # 3. 提取当前关节在所有时间步的特征
            joint_features = x[:, :, j].unsqueeze(2)  # [batch_size, seq_len, 1]
            
            # 4. 应用掩码矩阵进行加权聚合
            # 将掩码扩展到batch维度，然后进行批量矩阵乘法
            expanded_mask = mask_j.expand(batch_size, -1, -1)
            aggregated_joint = torch.bmm(expanded_mask, joint_features)
            
            # 5. 收集聚合后的关节特征
            aggregated_features.append(aggregated_joint)
        
        # 6. 拼接所有关节的聚合特征，恢复原始维度顺序
        aggregated_x = torch.cat(aggregated_features, dim=2)
        
        return aggregated_x
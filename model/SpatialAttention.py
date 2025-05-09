import torch
import torch.nn as nn
import numpy as np
import math

class SkeletonAwareSpatialAttention(nn.Module):
    """
    基于骨骼感知的空间自注意力机制
    
    该模块用于提取每一帧中关节之间的依赖关系，同时考虑人体骨骼的运动学先验信息。
    通过引入运动学邻接矩阵，模型能够同时考虑关节间的实际相关性与人体结构中的先验连接关系。
    """
    def __init__(self, joint_dim, d_model=128, num_heads=8, dropout=0.1):
        """
        初始化基于骨骼感知的空间自注意力模块
        
        参数:
            joint_dim: 关节特征维度
            d_model: 注意力模块的隐藏维度
            num_heads: 多头注意力的头数
            dropout: Dropout比率
        """
        super(SkeletonAwareSpatialAttention, self).__init__()
        
        self.joint_dim = joint_dim
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert self.head_dim * num_heads == d_model, "d_model必须能被num_heads整除"
        
        # 定义三个编码器，用于生成query、key和value
        self.q_encoder = nn.Linear(3, d_model)  # 每个关节是3D坐标
        self.k_encoder = nn.Linear(3, d_model)
        self.v_encoder = nn.Linear(3, d_model)
        
        # 输出投影层
        self.output_projection = nn.Linear(d_model, d_model)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 初始化运动学邻接矩阵
        self.register_buffer('kinematic_adjacency', self._init_kinematic_adjacency())
        
    def _init_kinematic_adjacency(self):
        """
        初始化运动学邻接矩阵，包含远端关节和对称关节的信息
        
        返回:
            kinematic_adjacency: 运动学邻接矩阵 [joint_dim, joint_dim]
        """
        # 这里简化处理，实际应用中需要根据具体的人体骨骼模型定义
        # 假设joint_dim=22，对应Human3.6M数据集的关节数量
        
        # 1. 初始化直接连接的邻接矩阵 (一级邻接关系)
        direct_adjacency = torch.zeros(self.joint_dim, self.joint_dim)
        
        # 定义人体骨骼的连接关系 (这里是简化示例，实际应用需要根据数据集调整)
        # 格式: (父关节索引, 子关节索引)
        skeleton_connections = [
            (0, 1), (1, 2), (2, 3),  # 脊柱到头部
            (0, 4), (4, 5), (5, 6),  # 右臂
            (0, 7), (7, 8), (8, 9),  # 左臂
            (0, 10), (10, 11), (11, 12),  # 右腿
            (0, 13), (13, 14), (14, 15),  # 左腿
        ]
        
        # 填充直接连接的邻接矩阵
        for parent, child in skeleton_connections:
            direct_adjacency[parent, child] = 1.0
            direct_adjacency[child, parent] = 1.0  # 双向连接
        
        # 2. 初始化对称关节的邻接矩阵
        symmetric_adjacency = torch.zeros(self.joint_dim, self.joint_dim)
        
        # 定义对称关节对 (这里是简化示例)
        symmetric_pairs = [
            (4, 7), (5, 8), (6, 9),  # 左右手臂对称
            (10, 13), (11, 14), (12, 15),  # 左右腿对称
        ]
        
        # 填充对称关节的邻接矩阵
        for joint1, joint2 in symmetric_pairs:
            symmetric_adjacency[joint1, joint2] = 0.8  # 对称关节的权重可以调整
            symmetric_adjacency[joint2, joint1] = 0.8
        
        # 3. 合并得到最终的运动学邻接矩阵: A = A_d + A_s
        kinematic_adjacency = direct_adjacency + symmetric_adjacency
        
        # 确保对角线上的值为1，表示自连接
        kinematic_adjacency = kinematic_adjacency + torch.eye(self.joint_dim)
        
        return kinematic_adjacency
    
    def forward(self, x):
        """
        前向传播函数
        
        参数:
            x: 输入序列 [batch_size, seq_len, joint_dim, 3]
                batch_size: 批量大小
                seq_len: 序列长度
                joint_dim: 关节数量
                3: 每个关节的3D坐标 (x, y, z)
            
        返回:
            output: 经过空间自注意力处理后的序列 [batch_size, seq_len, joint_dim, d_model]
        """
        batch_size, seq_len, joint_dim, _ = x.shape
        
        # 处理每一帧
        outputs = []
        for t in range(seq_len):
            # 提取当前帧
            frame = x[:, t]  # [batch_size, joint_dim, 3]
            
            # 生成query, key, value
            q = self.q_encoder(frame)  # [batch_size, joint_dim, d_model]
            k = self.k_encoder(frame)  # [batch_size, joint_dim, d_model]
            v = self.v_encoder(frame)  # [batch_size, joint_dim, d_model]
            
            # 分割多头
            q = q.view(batch_size, joint_dim, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            k = k.view(batch_size, joint_dim, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            v = v.view(batch_size, joint_dim, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            
            # 计算注意力分数 (公式11)
            attention_scores = torch.matmul(q, k.transpose(-2, -1))  # [batch_size, num_heads, joint_dim, joint_dim]
            
            # 添加运动学邻接矩阵 (公式12)
            # 扩展邻接矩阵到多头维度
            kinematic_adj_expanded = self.kinematic_adjacency.unsqueeze(0).unsqueeze(0)
            kinematic_adj_expanded = kinematic_adj_expanded.expand(batch_size, self.num_heads, -1, -1)
            
            # 添加运动学先验
            attention_scores = attention_scores + kinematic_adj_expanded
            
            # 缩放并应用softmax (公式13)
            attention_scores = attention_scores / math.sqrt(self.head_dim)
            attention_weights = torch.softmax(attention_scores, dim=-1)
            attention_weights = self.dropout(attention_weights)
            
            # 应用注意力权重
            context = torch.matmul(attention_weights, v)  # [batch_size, num_heads, joint_dim, head_dim]
            
            # 合并多头
            context = context.permute(0, 2, 1, 3).contiguous()  # [batch_size, joint_dim, num_heads, head_dim]
            context = context.view(batch_size, joint_dim, self.d_model)  # [batch_size, joint_dim, d_model]
            
            # 输出投影
            output = self.output_projection(context)  # [batch_size, joint_dim, d_model]
            
            outputs.append(output.unsqueeze(1))
        
        # 拼接所有帧的输出
        outputs = torch.cat(outputs, dim=1)  # [batch_size, seq_len, joint_dim, d_model]
        
        return outputs
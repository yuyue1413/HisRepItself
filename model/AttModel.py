from torch.nn import Module
from torch import nn
import torch
# import model.transformer_base
import math
from model import GCN
import utils.util as util
import numpy as np

from model.TemporalAggregation import LearnableMaskTemporalAggregation
from model.SpatialAttention import SkeletonAwareSpatialAttention


class AttModel(Module):

    def __init__(self, in_features=48, kernel_size=5, d_model=512, num_stage=2, dct_n=10):
        super(AttModel, self).__init__()

        self.kernel_size = kernel_size
        self.d_model = d_model
        self.dct_n = dct_n
        assert kernel_size == 10

        # 可学习的时间聚合模块 - 用于自适应地强化重要关节并削弱冗余关节
        self.temporal_aggregation = LearnableMaskTemporalAggregation(seq_len=50, joint_dim=in_features)

        # 定义映射函数 f_k 和 f_q，用于生成 key 和 query
        # f_k: 从历史运动 X_{1:L} 映射得到 key
        self.convK = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=d_model, kernel_size=6,
                                             bias=False),
                                   nn.ReLU(),
                                   nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5,
                                             bias=False),
                                   nn.ReLU())

        # f_q: 从当前运动 X_{L+1:N} 映射得到 query
        self.convQ = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=d_model, kernel_size=6,
                                             bias=False),
                                   nn.ReLU(),
                                   nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5,
                                             bias=False),
                                   nn.ReLU())

        # 新增: 基于骨骼感知的空间自注意力模块
        # 假设每个关节有3个坐标，关节数量为in_features/3
        self.joint_num = in_features // 3
        self.spatial_attention = SkeletonAwareSpatialAttention(
            joint_dim=self.joint_num,
            d_model=128,  # 空间注意力的隐藏维度
            num_heads=8,  # 多头注意力的头数
            dropout=0.1
        )

        # 空间注意力输出的特征映射层
        self.spatial_projection = nn.Linear(128 * self.joint_num, d_model)

        # 空间注意力模块 - 用于捕捉关节间的空间依赖关系
        # 这里使用GCN来实现空间注意力
        self.gcn = GCN.GCN(input_feature=dct_n * 2 + d_model,
                           hidden_feature=d_model,
                           p_dropout=0.3,
                           num_stage=num_stage,
                           node_n=in_features)

    def forward(self, src, output_n=25, input_n=50, itera=1):
        """
        :param src: [batch_size, seq_len, feat_dim]  (batch_size, input_n 50 + output_n 10, 66)
        :param output_n: 要预测的未来帧数
        :param input_n: 用于预测的过去帧数
        :param itera: 迭代次数，若设置大于1，会执行递归预测（自回归）
        :return:
        """
        dct_n = self.dct_n
        src = src[:, :input_n]  # [bs, in_n, dim] 截取输入序列前input_n帧

        # 1. 应用可学习的时间聚合模块 - 自适应地强化重要关节并削弱冗余关节
        src = self.temporal_aggregation(src)

        src_tmp = src.clone()  # 输入序列  (batch_size, 50, 66)
        bs = src.shape[0]  # batch_size

        # 2. 将N帧人体姿势划分为两个子集：历史运动和当前运动
        # 历史运动: X_{1:L}，当前运动: X_{L+1:N}
        # 这里L = input_n - output_n - kernel_size

        # 历史运动用于生成key  使用前input_n - output_n = 40帧
        src_key_tmp = src_tmp.transpose(1, 2)[:, :, :(input_n - output_n)].clone()

        # 当前运动用于生成query   使用后kernel_size = 10帧
        src_query_tmp = src_tmp.transpose(1, 2)[:, :, -self.kernel_size:].clone()

        # 3. 准备DCT变换矩阵，用于在轨迹空间中编码人体运动的时间信息
        dct_m, idct_m = util.get_dct_matrix(self.kernel_size + output_n)
        dct_m = torch.from_numpy(dct_m).float().cuda()
        idct_m = torch.from_numpy(idct_m).float().cuda()

        # 4. 生成value - 在轨迹空间中表示人体关节的轨迹
        vn = input_n - self.kernel_size - output_n + 1  # value的数量
        vl = self.kernel_size + output_n  # 每个value的长度

        # 构建滑动窗口索引，用于提取轨迹片段
        idx = np.expand_dims(np.arange(vl), axis=0) + \
              np.expand_dims(np.arange(vn), axis=1)

        # 提取轨迹片段并应用DCT变换，得到value
        src_value_tmp = src_tmp[:, idx].clone().reshape(
            [bs * vn, vl, -1])
        src_value_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), src_value_tmp).reshape(
            [bs, vn, dct_n, -1]).transpose(2, 3).reshape(
            [bs, vn, -1])  # [bs, vn, 66*dct_n]

        # 5. 准备当前运动的索引，用于后续处理
        idx = list(range(-self.kernel_size, 0, 1)) + [-1] * output_n
        outputs = []

        # 6. 生成key - 从历史运动映射得到
        key_tmp = self.convK(src_key_tmp / 1000.0)  # 归一化输入

        # 7. 迭代预测
        for i in range(itera):
            # 生成query - 从当前运动映射得到
            query_tmp = self.convQ(src_query_tmp / 1000.0)  # 归一化输入

            # 计算注意力得分 - 公式(4)
            score_tmp = torch.matmul(query_tmp.transpose(1, 2), key_tmp) + 1e-15

            # 归一化注意力权重
            att_tmp = score_tmp / (torch.sum(score_tmp, dim=2, keepdim=True))

            # 计算时间注意力的输出 - 公式(5)
            dct_att_tmp = torch.matmul(att_tmp, src_value_tmp)[:, 0].reshape(
                [bs, -1, dct_n])

            # 8. 提取当前运动的DCT系数 c_c
            input_gcn = src_tmp[:, idx]
            dct_in_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), input_gcn).transpose(1, 2)

            # 9. 新增: 应用基于骨骼感知的空间自注意力
            # 重塑输入以适应空间自注意力模块
            # 将关节特征重组为[batch_size, seq_len, joint_num, 3]格式
            spatial_input = src_tmp.reshape(bs, -1, self.joint_num, 3)

            # 应用空间自注意力
            spatial_features = self.spatial_attention(spatial_input)  # [bs, seq_len, joint_num, 128]

            # 重塑空间特征以便后续处理
            spatial_features = spatial_features.reshape(bs, -1, self.joint_num * 128)
            # 投影到相同的维度空间
            spatial_features = self.spatial_projection(spatial_features[:, 0])  # [bs, d_model]

            # 调整维度以匹配其他特征
            spatial_features = spatial_features.unsqueeze(1).expand(-1, dct_in_tmp.shape[1], -1)

            # 10. 拼接时间注意力输出、空间注意力输出和当前运动的DCT系数 - 公式(6)
            # dct_in_tmp相当于c_c，dct_att_tmp相当于u_t，spatial_features相当于u_s
            dct_in_tmp = torch.cat([dct_in_tmp, dct_att_tmp, spatial_features],
                                   dim=-1)  # shape: torch.Size([2, 66, 296])
            # dct_in_tmp = torch.cat([dct_in_tmp, dct_att_tmp], dim=-1)  # shape: torch.Size([2, 66, 40])

            # print("shape:", dct_in_tmp.shape)

            # 11. 应用GCN进行空间建模，进一步融合特征
            dct_out_tmp = self.gcn(dct_in_tmp)  # 132*296   40*256
            # 12. 应用IDCT变换，将频域特征转回时域，得到预测的人体姿态
            out_gcn = torch.matmul(idct_m[:, :dct_n].unsqueeze(dim=0),
                                   dct_out_tmp[:, :, :dct_n].transpose(1, 2))
            outputs.append(out_gcn.unsqueeze(2))
            # 13. 如果需要多次迭代预测，更新输入序列
            if itera > 1:
                # 更新序列，将预测结果添加到输入序列末尾
                out_tmp = out_gcn.clone()[:, 0 - output_n:]
                src_tmp = torch.cat([src_tmp, out_tmp], dim=1)

                # 更新key-value和query
                vn = 1 - 2 * self.kernel_size - output_n
                vl = self.kernel_size + output_n
                idx_dct = np.expand_dims(np.arange(vl), axis=0) + \
                          np.expand_dims(np.arange(vn, -self.kernel_size - output_n + 1), axis=1)

                # 更新key
                src_key_tmp = src_tmp[:, idx_dct[0, :-1]].transpose(1, 2)
                key_new = self.convK(src_key_tmp / 1000.0)
                key_tmp = torch.cat([key_tmp, key_new], dim=2)

                # 更新value
                src_dct_tmp = src_tmp[:, idx_dct].clone().reshape(
                    [bs * self.kernel_size, vl, -1])
                src_dct_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), src_dct_tmp).reshape(
                    [bs, self.kernel_size, dct_n, -1]).transpose(2, 3).reshape(
                    [bs, self.kernel_size, -1])
                src_value_tmp = torch.cat([src_value_tmp, src_dct_tmp], dim=1)

                # 更新query
                src_query_tmp = src_tmp[:, -self.kernel_size:].transpose(1, 2)

        # 14. 拼接所有迭代的输出
        outputs = torch.cat(outputs, dim=2)
        return outputs
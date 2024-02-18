from random import random
#
import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp

from HGTAN.Attn_head import Attn_head

# SEED = random.randint(0, 1000000)  # 生成一个随机的种子
# print("hyperedge_attn Seed:", SEED)
# torch.cuda.manual_seed_all(SEED)
class hyperedge_atten(torch.nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(hyperedge_atten, self).__init__()
        self.intra_hpyeredge = Attn_head(nfeat, nhid, in_drop=0, coef_drop=0, activation=nn.ELU(), residual=True)
        self.dropout = dropout
        self.regression_layer = nn.Linear(nhid, 1)

    # def forward(self, x, H):
    #     # x形状为[1329,1]
    #     # H的形状为[1,1329,200]
    #     # 批次大小和节点数相同
    #     num_nodes = x.size(0)
    #
    #     # 确保x的维度为 [1, num_nodes, nfeat]
    #     x = x.unsqueeze(0)
    #
    #     hyperedge_fts = torch.randn(1, 0, num_nodes).cuda()
    #
    #     num_hyperedges = H.size(2)
    #
    #     for i in range(num_hyperedges):
    #         # 选取属于当前超边的所有节点
    #         intra_hyperedge_fts = torch.randn(1, 0, num_nodes).cuda()
    #         node_indices = torch.nonzero(H[:, :, i], as_tuple=True)[1]
    #         selected_node_features = x[:, node_indices, :]
    #
    #         # 如果没有节点属于当前超边，则跳过
    #         if selected_node_features.size(1) == 0:
    #             continue
    #
    #         # 将节点特征扩展到与 intra_hyperedge_fts 相同的形状
    #         expanded_features = selected_node_features.expand(-1, -1, 1329)
    #
    #
    #         # 现在可以进行拼接，因为非拼接维度的大小已经匹配
    #         intra_hyperedge_fts = torch.cat([intra_hyperedge_fts, expanded_features], dim=1)
    #
    #         after_intra = self.intra_hpyeredge(intra_hyperedge_fts)
    #         pooling = torch.nn.MaxPool1d(intra_hyperedge_fts.size(1), stride=1)
    #         e_fts = pooling(after_intra.permute(0, 2, 1))
    #         hyperedge_fts = torch.cat([hyperedge_fts, e_fts.permute(0, 2, 1)], dim=1)
    #
    #     # 首先，我们沿着超边特征的维度聚合特征
    #     aggregated_features = torch.mean(hyperedge_fts, dim=1)  # 使用平均聚合，形状变为 [1, 1329]
    #     # 调整形状以适应全连接层
    #     aggregated_features = aggregated_features.squeeze(0)  # 移除批次维度，形状变为 [1329]
    #     # 应用全连接层
    #     regression_outputs = self.regression_layer(aggregated_features)  # 现在形状是 [1329, 1]
    #     regression_outputs = regression_outputs.view(-1, 1)
    #     return regression_outputs
    def forward(self, x, H):
        # x形状为[1329,1]
        # H的形状为[1,1329,200]
        # 批次大小和节点数相同
        num_nodes = x.size(0)
        x = x.unsqueeze(0)

        # 初始化节点特征汇总张量
        node_features_aggregated = torch.zeros(1, num_nodes, 2000).cuda()

        num_hyperedges = H.size(2)

        # 预先计算所有超边的节点索引
        precomputed_indices = [torch.nonzero(H[:, :, i], as_tuple=True)[1] for i in range(num_hyperedges)]

        for i in range(num_hyperedges):
            node_indices = precomputed_indices[i]
            if node_indices.size(0) == 0:
                continue

            intra_hyperedge_fts = x[:, node_indices, :]
            after_intra = self.intra_hpyeredge(intra_hyperedge_fts)

            node_features_aggregated[:, node_indices, :] += after_intra

        # 为每个节点生成预测
        regression_outputs = self.regression_layer(node_features_aggregated.squeeze())
        return regression_outputs.view(-1, 1)

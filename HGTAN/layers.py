import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from HGTAN.AttenLayer import AttentionLayer


# class HGNN_conv(nn.Module):
#     def __init__(self, in_ft, out_ft, bias=True):
#         super(HGNN_conv, self).__init__()
#
#         self.weight = Parameter(torch.Tensor(in_ft, out_ft))
#         if bias:
#             self.bias = Parameter(torch.Tensor(out_ft))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.weight.size(1))
#         self.weight.data.uniform_(-stdv, stdv)
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)
#
#     def forward(self, x: torch.Tensor, G: torch.Tensor):
#         x = x.matmul(self.weight)
#         if self.bias is not None:
#             x = x + self.bias
#
#         # 假设G[0]包含超边索引，G[1]包含节点索引
#         edge_indices = G[0]
#         node_indices = G[1]
#
#         # 创建一个空的特征矩阵来聚合超边信息
#         # 正确使用torch.zeros创建零矩阵
#         edge_features = torch.zeros((int(edge_indices.max().item() + 1), x.size(1)),
#                                     dtype=x.dtype, device=x.device)
#
#         # 将节点特征累加到相应的超边上
#         edge_features.index_add_(0, edge_indices, x[node_indices])
#
#         # 重新分配特征回节点
#         # 使用边索引的方式来重新聚合节点特征
#         node_features = torch.zeros_like(x)
#         node_features.scatter_add_(0, node_indices.unsqueeze(1).expand(-1, x.size(1)), edge_features[edge_indices])
#
#         return node_features
class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        self.att_layer = AttentionLayer(in_ft)
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, G):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias

        edge_indices = G[0]
        node_indices = G[1]

        attention = self.att_layer(x, edge_indices)

        # 初始化节点特征矩阵
        node_features = torch.zeros_like(x)

        # 计算通过注意力加权的特征
        weighted_features = attention * x[edge_indices]

        # 调整 node_indices 的形状以匹配 weighted_features
        node_indices_expanded = node_indices.unsqueeze(1).expand_as(weighted_features)

        # 使用 scatter_add_ 在节点特征矩阵中累加加权特征
        node_features.scatter_add_(0, node_indices_expanded, weighted_features)

        return node_features
        # # 使用注意力权重更新节点特征
        # node_features = torch.zeros_like(x)
        # for i in range(G.shape[1]):  # 使用 G.shape[1] 而不是 edge_indices.size(1)
        #     # 现在可以安全地使用 i 来索引 edge_indices 和 node_indices
        #     node_features[node_indices[i]] += attention[i] * x[edge_indices[i]]
        #
        # return node_features
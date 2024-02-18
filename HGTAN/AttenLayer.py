import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math

class AttentionLayer(nn.Module):
    def __init__(self, in_features):
        super(AttentionLayer, self).__init__()
        self.in_features = in_features
        self.att_weight = Parameter(torch.Tensor(32, in_features))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.att_weight.size(1))
        self.att_weight.data.uniform_(-stdv, stdv)

    def forward(self, x, hyperedge_index_np):
        # 使用边索引将节点特征聚合到每条边
        hyperedge_features = x[hyperedge_index_np]

        # 计算注意力权重
        weights = torch.matmul(hyperedge_features, self.att_weight.t())
        attention = F.softmax(weights, dim=1)

        return attention

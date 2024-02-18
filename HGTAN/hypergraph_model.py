import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv
from Attn_head import Attn_head


class HypergraphAttentionNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, attn_out_sz, attn_in_drop=0.0, attn_coef_drop=0.0,
                 attn_activation=F.relu, attn_residual=False):
        super(HypergraphAttentionNetwork, self).__init__()
        self.hypergraph_conv = HypergraphConv(in_channels, 32)
        self.attn_head = Attn_head(32, attn_out_sz, in_drop=attn_in_drop, coef_drop=attn_coef_drop,
                                   activation=attn_activation, residual=attn_residual)
        self.fc = nn.Linear(attn_out_sz, 1)  # 新增加的全连接层

    def forward(self, x, hyperedge_index):
        x = F.relu(self.hypergraph_conv(x, hyperedge_index))
        # x是二维的，在第二维添加一个维度
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.attn_head(x)
        x = self.fc(x)  # 应用全连接层
        return x.squeeze(2)  # 去除多余的维度

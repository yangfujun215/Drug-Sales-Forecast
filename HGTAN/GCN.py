import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GNNModel(nn.Module):
    def __init__(self, num_features, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        # 图卷积层
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        # 添加一个额外的全连接层来调整输出
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        # # 应用第一个图卷积层和ReLU激活函数
        # x = F.relu(self.conv1(x, edge_index))
        # # 应用第二个图卷积层
        # x = self.conv2(x, edge_index)
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        # 应用全连接层
        x = self.fc(x)
        return x

# 假设 num_features 是 GRU 输出的特征数量
# hidden_dim 是图神经网络隐藏层的维度
# output_dim 是预测结果的维度

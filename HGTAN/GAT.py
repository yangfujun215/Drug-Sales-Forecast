import torch
from torch_geometric.nn import GATConv
import torch.nn.functional as F


class GAT(torch.nn.Module):
    def __init__(self, num_features, hidden_channels):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_features, hidden_channels, heads=4, dropout=0)
        self.conv2 = GATConv(hidden_channels * 4, 1, heads=4, dropout=0)  # 假设输出是单值
        self.fc = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.fc(x)  # 应用全连接层
        return x

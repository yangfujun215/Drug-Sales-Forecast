from torch import nn

import torch.nn.functional as F

from HGTAN.layers import HGNN_conv


class HGNN(nn.Module):
    def __init__(self, in_channels, n_hid, dropout=0):
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_channels, n_hid)
        # self.hgc2 = HGNN_conv(n_hid, n_hid)
        # 添加一个全连接层，输出维度为1（或者是你回归任务中需要的维度）
        self.fc = nn.Linear(n_hid, 1)

    def forward(self, x, hyperedge_index_np):
        x = F.relu(self.hgc1(x, hyperedge_index_np))
        # 通过全连接层得到最终的回归预测结果
        x = self.fc(x)
        return x

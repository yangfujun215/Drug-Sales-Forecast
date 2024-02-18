import torch
import torch.nn as nn
import random

# SEED = random.randint(0, 1000000)  # 生成一个随机的种子
# print("Attn_head Seed:", SEED)
# torch.cuda.manual_seed_all(SEED)


class Attn_head(nn.Module):
    def __init__(self,
                 in_channel,
                 out_sz,
                 in_drop=0.0,  # 输入系数的dropout率
                 coef_drop=0.0,  # 注意力系数的dropout率
                 activation=None,
                 residual=False):
        super(Attn_head, self).__init__()

        self.in_channel = in_channel
        self.out_sz = out_sz
        self.in_drop = in_drop
        self.coef_drop = coef_drop
        self.activation = activation
        self.residual = residual

        self.conv1 = nn.Conv1d(self.in_channel, self.out_sz, 1)
        self.conv2_1 = nn.Conv1d(self.out_sz, 1, 1)
        self.conv2_2 = nn.Conv1d(self.out_sz, 1, 1)
        self.leakyrelu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)
        self.in_dropout = nn.Dropout()
        self.coef_dropout = nn.Dropout()
        self.res_conv = nn.Conv1d(self.in_channel, self.out_sz, 1)

    def forward(self, x):

        seq = x.permute(0, 2, 1)

        seq_fts = self.conv1(seq)

        f_1 = self.conv2_1(seq_fts)
        f_2 = self.conv2_2(seq_fts)
        logits = f_1 + torch.transpose(f_2, 2, 1)
        logits = self.leakyrelu(logits)
        coefs = self.softmax(logits)

        if self.coef_drop != 0.0:
            coefs = self.coef_dropout(coefs)
        if self.in_dropout != 0.0:
            seq_fts = self.in_dropout(seq_fts)
        # print("coefs shape:", coefs.shape)
        # print("seq_fts shape before transpose:", seq_fts.shape)
        seq_fts_transposed = torch.transpose(seq_fts, 2, 1)  # [1329, 1, 32]
        ret = torch.matmul(coefs, seq_fts_transposed)

        if self.residual:
            if seq.shape[1] != ret.shape[2]:
                ret = ret + self.res_conv(seq).permute(0, 2, 1)
            else:
                ret = ret + seq.permute(0, 2, 1)

        return self.activation(ret)

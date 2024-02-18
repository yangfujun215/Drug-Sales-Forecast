import numpy as np
import pandas as pd
import torch
import random
# SEED = random.randint(0, 1000000)  # 生成一个随机的种子
# print("load_H Seed:", SEED)
# np.random.seed(SEED)
def get_fund_adj_H_from_new_file(new_file_path):
    # 初始化张量，尺寸为0x1329x200
    tensor = np.random.rand(0, 2000, 150)

    # 读取新文件
    df = pd.read_csv(new_file_path, header=None)
    fund_adj = df.values

    # 将文件内容转换为张量并与原始张量连接
    tensor = np.concatenate((tensor, fund_adj[None]), axis=0)
    tensor = torch.from_numpy(tensor)
    return tensor


# 使用新文件的路径
H = get_fund_adj_H_from_new_file('../data/A_adj(2000150).csv')
print(H.shape)
print(H)

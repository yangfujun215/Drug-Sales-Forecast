# from scipy.sparse import coo_matrix
# from torch_geometric.utils import from_scipy_sparse_matrix
# import networkx as nx
# import pandas as pd
#
# # 加载邻接矩阵
# A = pd.read_csv('../data/A_adj(13291329).csv', header=None).values
#
# # 将邻接矩阵转换为图形
# # 将邻接矩阵转换为 COO 格式的稀疏矩阵
# A_sparse = coo_matrix(A)
#
# # 转换为 PyTorch Geometric 的 edge_index
# edge_index, _ = from_scipy_sparse_matrix(A_sparse)
# print(edge_index.shape)
import pandas as pd
import torch

# 加载邻接矩阵文件
file_path = '../data/A_adj(13291329).csv'
adj_matrix = pd.read_csv(file_path, header=None)

# 将邻接矩阵转换为 PyTorch 张量
adj_tensor = torch.tensor(adj_matrix.values)

# 找出所有非零元素的位置（即存在边的位置）
edge_index = adj_tensor.nonzero(as_tuple=False).t()

print(edge_index.shape, edge_index)

import torch
import pandas as pd

# 加载超边与节点的关系文件
hyperedge_node_path = '../data/A_adj(2000150).csv'
hyperedge_node_matrix = pd.read_csv(hyperedge_node_path, header=None).values
# 转换为超边索引
num_nodes, num_hyperedges = hyperedge_node_matrix.shape
rows, cols = torch.where(torch.tensor(hyperedge_node_matrix, dtype=torch.bool))
# 构建 hyperedge_index
# 形状为 [2, num_edges], 其中 num_edges 是超边中包含的节点对的总数
hyperedge_index = torch.stack([rows, cols], dim=0)
# hyperedge_index = hyperedge_index.view(1329, -1).float()
print("Shape:", hyperedge_index.shape)
print("Type:", hyperedge_index.dtype)
print("Max index:", hyperedge_index.max())
print("Min index:", hyperedge_index.min())

if (hyperedge_index < 0).any():
    print("Error: Negative index found in hyperedge_index.")

# 设置张量的目标维度大小
if (hyperedge_index >= 2000).any():
    print(f"Error: Index out of bounds in hyperedge_index. Max allowed index: {2000-1}")

if not torch.is_tensor(hyperedge_index) or hyperedge_index.dtype != torch.long:
    print("Error: hyperedge_index must be a tensor of type torch.long")

# 现在hyperedge_index节点与超边的连接可以直接输入到超图中进行计算。
# 每一列代表一个节点与一个超边的关系。
# 第一行 (hyperedge_index[0, :]) 包含节点的索引。
# 第二行 (hyperedge_index[1, :]) 包含对应的超边索引。
# 现在 hyperedge_index 可以用于超图神经网络的计算


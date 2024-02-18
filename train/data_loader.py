# import torch
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn.model_selection import train_test_split
#
#
# def create_sliding_windows(data, window_size, horizon):
#     X, y = [], []
#     for i in range(len(data) - window_size - horizon + 1):
#         X.append(data[i:i + window_size])
#         y.append(data[i + window_size + horizon - 1])
#     return np.array(X), np.array(y)
#
#
# # 读取数据
# data = pd.read_csv('../data/input011.csv')
# data.columns = ['Date', 'Drug_ID', 'Drug_Category', 'Sales']
# data['Date'] = pd.to_datetime(data['Date'])
# data.sort_values(['Drug_ID', 'Date'], inplace=True)
#
# # 设置窗口大小和预测范围
# window_size = 5
# horizon = 1
#
# # 存储每个药品的窗口和目标
# all_drug_windows = []
# all_drug_targets = []
#
# # 初始化数组用于存储数据
# all_train_windows = []
# all_val_windows = []
# all_test_windows = []
# all_train_targets = []
# all_val_targets = []
# all_test_targets = []
#
# # 定义划分比例
# total_days = 1155
# train_ratio = 0.8
# val_ratio = 0.1
# test_ratio = 0.1
# # 计算划分索引
# train_size = int(total_days * train_ratio)
# val_size = int(total_days * val_ratio)
# test_size = total_days - train_size - val_size
#
# # 创建滑动窗口
# for _, group in data.groupby('Drug_ID'):
#     sales = group['Sales'].values
#     scaler = MinMaxScaler()
#     normalized_sales = scaler.fit_transform(sales.reshape(-1, 1)).flatten()
#     X, y = create_sliding_windows(normalized_sales, window_size, horizon)
#     all_drug_windows.append(X)
#     all_drug_targets.append(y)
#
#     # 统一划分数据集
#     X_train, y_train = X[:train_size], y[:train_size]
#     X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
#     X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]
#
#     all_train_windows.append(X_train)
#     all_val_windows.append(X_val)
#     all_test_windows.append(X_test)
#     all_train_targets.append(y_train)
#     all_val_targets.append(y_val)
#     all_test_targets.append(y_test)
#
#
# # 重组数据以匹配GRU的输入形状
# def reshape_and_concatenate(data_list):
#     # 确保所有药品窗口数量相同
#     num_windows_per_drug = min(len(windows) for windows in data_list)
#
#     # 重塑数据
#     batched_data = np.array([drug_windows[:num_windows_per_drug] for drug_windows in data_list])
#     return batched_data.transpose(1, 0, 2).reshape(-1, window_size, 1)
#
#
# # 重塑并连接数据
# train_features = reshape_and_concatenate(all_train_windows)
# val_features = reshape_and_concatenate(all_val_windows)
# test_features = reshape_and_concatenate(all_test_windows)
# train_targets = np.concatenate(all_train_targets, axis=0)
# val_targets = np.concatenate(all_val_targets, axis=0)
# test_targets = np.concatenate(all_test_targets, axis=0)
#
# # 转换为张量
# train_features_tensor = torch.tensor(train_features, dtype=torch.float32)
# val_features_tensor = torch.tensor(val_features, dtype=torch.float32)
# test_features_tensor = torch.tensor(test_features, dtype=torch.float32)
# train_targets_tensor = torch.tensor(train_targets, dtype=torch.float32)
# val_targets_tensor = torch.tensor(val_targets, dtype=torch.float32)
# test_targets_tensor = torch.tensor(test_targets, dtype=torch.float32)
#
# print(test_features_tensor.shape)
# print(test_targets_tensor.shape)
#
# # 创建TensorDataset
# train_dataset = TensorDataset(train_features_tensor, train_targets_tensor)
# val_dataset = TensorDataset(val_features_tensor, val_targets_tensor)
# test_dataset = TensorDataset(test_features_tensor, test_targets_tensor)
#
# # 创建DataLoader
# train_loader = DataLoader(train_dataset, batch_size=1329, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=1329, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=1329, shuffle=False)
import random

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import torch

# SEED = random.randint(0, 1000000)  # 生成一个随机的种子
# print("data_loader Seed:", SEED)
# np.random.seed(SEED)
def create_sliding_windows(data, window_size, horizon):
    X, y = [], []
    for i in range(len(data) - window_size - horizon + 1):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size + horizon - 1])
    return np.array(X), np.array(y)


# 读取数据
data = pd.read_csv('../data/input0107.csv')
data.columns = ['Date', 'Drug_ID', 'Sales', 'Drug_Category', 'Drug_SubCategory']
data['Date'] = pd.to_datetime(data['Date'])
data.sort_values(['Drug_ID', 'Date'], inplace=True)

# 设置窗口大小和预测范围
window_size = 10
horizon = 1

# 存储每个药品的窗口和目标
all_drug_windows = []
all_drug_targets = []




scalers = {}  # 用于存储每个药品的归一化器
# 遍历每个药品，创建滑动窗口
for drug_id, group in data.groupby('Drug_ID'):
    # print(group)
    # print("Original sales data:", group['Sales'].head())  # 打印前几个销量数据
    group['Sales_Log'] = np.log(group['Sales'] + 1)
    scaler = MinMaxScaler()
    normalized_sales = scaler.fit_transform(group['Sales_Log'].values.reshape(-1, 1)).flatten()
    scalers[drug_id] = scaler  # 存储归一化器
    print("Normalized sales data:", normalized_sales[:5])  # 打印归一化后的前几个数据
    X, y = create_sliding_windows(normalized_sales, window_size, horizon)
    # print("Window data sample:", X[0])  # 打印一个窗口样本
    # print("Target data sample:", y[0])  # 打印对应的目标样本
    all_drug_windows.append(X)
    all_drug_targets.append(y)

# 确定所有药品的最小窗口数量
min_windows = min(len(windows) for windows in all_drug_windows)

# 定义划分比例
train_ratio = 0.9
val_ratio = 0
test_ratio = 0.1

# 计算划分索引
train_size = int(min_windows * train_ratio)
val_size = int(min_windows * val_ratio)
test_size = min_windows - train_size - val_size

print("Train Size:", train_size)
print("Validation Size:", val_size)
print("Test Size:", test_size)

# 统一划分每个药品的数据集
all_train_windows, all_val_windows, all_test_windows = [], [], []
all_train_targets, all_val_targets, all_test_targets = [], [], []

for windows, targets in zip(all_drug_windows, all_drug_targets):
    X_train, y_train = windows[:train_size], targets[:train_size]
    X_val, y_val = windows[train_size:train_size + val_size], targets[train_size:train_size + val_size]
    X_test, y_test = windows[-test_size:], targets[-test_size:]

    all_train_windows.append(X_train)
    all_val_windows.append(X_val)
    all_test_windows.append(X_test)
    all_train_targets.append(y_train)
    all_val_targets.append(y_val)
    all_test_targets.append(y_test)


def reshape_and_concatenate(data_list):
    reshaped_data = []
    for i in range(len(data_list[0])):
        for windows in data_list:
            reshaped_data.append(windows[i])
    return np.array(reshaped_data)


# 重塑并连接数据
train_features = reshape_and_concatenate(all_train_windows).reshape(-1, window_size, 1)
val_features = reshape_and_concatenate(all_val_windows).reshape(-1, window_size, 1)
test_features = reshape_and_concatenate(all_test_windows).reshape(-1, window_size, 1)
train_targets = reshape_and_concatenate(all_train_targets).reshape(-1, 1)
val_targets = reshape_and_concatenate(all_val_targets).reshape(-1, 1)
test_targets = reshape_and_concatenate(all_test_targets).reshape(-1, 1)
# print(train_features)
# 转换为张量
train_features_tensor = torch.tensor(train_features, dtype=torch.float32)
val_features_tensor = torch.tensor(val_features, dtype=torch.float32)
test_features_tensor = torch.tensor(test_features, dtype=torch.float32)
train_targets_tensor = torch.tensor(train_targets, dtype=torch.float32)
val_targets_tensor = torch.tensor(val_targets, dtype=torch.float32)
test_targets_tensor = torch.tensor(test_targets, dtype=torch.float32)

train_dataset = TensorDataset(train_features_tensor, train_targets_tensor)
val_dataset = TensorDataset(val_features_tensor, val_targets_tensor)
test_dataset = TensorDataset(test_features_tensor, test_targets_tensor)

# 创建DataLoader
batch_size = 2000  # 药品数量
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# num_batches = len(train_loader)
# print(f"Total number of batches: {num_batches}")
#
# # 检查一个批次的数据
# for batch in train_loader:
#     gru_inputs, targets = batch[0], batch[1]
#     print(f"GRU inputs shape: {gru_inputs.shape}")  # 应该是 (1329, window_size, 1)
#     print(f"Targets shape: {targets.shape}")        # 应该是 (1329, 1) 或者相应的形状
#     break  # 只查看第一个批次

# 假设 train_loader 是您已经创建好的 DataLoader

# def reshape_and_stack(data_list, window_size):
#     # 将每个药品的数据重塑为(num_windows, 1, window_size)
#     reshaped_data = [np.reshape(windows, (-1, 1, window_size)) for windows in data_list]
#     # 按窗口堆叠所有药品的数据，形成一个大数组，尺寸为(num_windows, num_drugs, window_size)
#     stacked_data = np.concatenate(reshaped_data, axis=1)
#     return stacked_data
#
#
# # 应用重塑和堆叠函数
# train_features = reshape_and_stack(all_train_windows, window_size)
# val_features = reshape_and_stack(all_val_windows, window_size)
# test_features = reshape_and_stack(all_test_windows, window_size)
#
# train_targets = reshape_and_stack(all_train_targets, 1)
# val_targets = reshape_and_stack(all_val_targets, 1)
# test_targets = reshape_and_stack(all_test_targets, 1)
#
# # 转换为张量
# train_features_tensor = torch.tensor(train_features, dtype=torch.float32)
# val_features_tensor = torch.tensor(val_features, dtype=torch.float32)
# test_features_tensor = torch.tensor(test_features, dtype=torch.float32)
# train_targets_tensor = torch.tensor(train_targets, dtype=torch.float32)
# val_targets_tensor = torch.tensor(val_targets, dtype=torch.float32)
# test_targets_tensor = torch.tensor(test_targets, dtype=torch.float32)
#
# # 创建 DataLoader
# train_loader = DataLoader(TensorDataset(train_features_tensor, train_targets_tensor), batch_size=64, shuffle=False)
# val_loader = DataLoader(TensorDataset(val_features_tensor, val_targets_tensor), batch_size=64, shuffle=False)
# test_loader = DataLoader(TensorDataset(test_features_tensor, test_targets_tensor), batch_size=64, shuffle=False)

# for batch in train_loader:
#     # 解包批次数据
#     features, targets = batch
#
#     # 打印特征和目标
#     print("Features: \n", features)
#     print("Targets: \n", targets)
#
#     # 只输出第一个批次的数据后退出循环
#     break

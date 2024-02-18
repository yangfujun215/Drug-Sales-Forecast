import torch
import torch.nn as nn
import random

# class GRUModel(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
#         super(GRUModel, self).__init__()
#         self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
#         # 添加一个全连接层来生成最终的输出
#         self.fc = nn.Linear(hidden_dim, output_dim)
#
#     def forward(self, x):
#         output, _ = self.gru(x)
#         # 取最后一个时间步的输出
#         last_step_output = output[:, -1, :]
#         # 通过全连接层生成最终的输出

#         return predicted_output


#
# try:
#     for batch in train_loader:
#         gru_inputs, targets = batch[0], batch[1]
#         gru_outputs = gru_model(gru_inputs)
#         # hypergraph_inputs = gru_outputs
#         # hyperedge_index = hyperedge_index
#         # hypergraph_outputs = hypergraph_model(hypergraph_inputs, hyperedge_index)
#         # print(f"Hypergraph outputs shape: {hypergraph_outputs.shape}")
#         print(f"GRU outputs shape: {gru_outputs.shape}")
# except Exception as e:
#     print(f"Error during GRU model forwarding: {e}")

# for batch in train_loader:
#     gru_inputs, targets = batch[0], batch[1]
#     print(f"GRU Inputs Shape: {gru_inputs.shape}, Type: {gru_inputs.dtype}")
#     print(f"Targets Shape: {targets.shape}, Type: {targets.dtype}")
#     break  # 只检查第一个批次
#
# try:
#     # 尝试通过GRU模型
#     gru_outputs = gru_model(gru_inputs)
#     print(f"GRU Outputs Shape: {gru_outputs.shape}")
#
#     # 将GRU的输出传递给超图神经网络模型
#     hypergraph_inputs = gru_outputs
#     hyperedge_index = hyperedge_index
#     hypergraph_outputs = hypergraph_model(hypergraph_inputs, hyperedge_index)
#     print(f"Hypergraph Outputs Shape: {hypergraph_outputs.shape}")
# except Exception as e:
#     print(f"Error during model forwarding: {e}")

# 测试 GRU 模型
# try:
#     test_input = torch.randn(1329, 5, 1)  # 假设的输入形状
#     test_output = gru_model(test_input)
#     print("GRU model forward pass successful")
# except Exception as e:
#     print(f"Error in GRU model: {e}")
#
# # 创建模拟的 GRU 输出
# gru_outputs = torch.randn(1329, 1)
#
# # 如果您的超图神经网络模型需要超边索引作为输入，创建一个模拟的超边索引
# # 假设每个样本有2个连接的超边
#
#
# try:
#     # 将测试数据传递给超图神经网络模型
#     hypergraph_outputs = hypergraph_model(gru_outputs, hyperedge_index)
#     print(f"Hypergraph model forward pass successful. Output shape: {hypergraph_outputs.shape}")
# except Exception as e:
#     print(f"Error in hypergraph model: {e}")


# # 检查 DataLoader 输出
# def check_dataloader_output(data_loader, num_batches_to_check):
#     for i, (inputs, targets) in enumerate(data_loader):
#         if i >= num_batches_to_check:
#             break
#         print(f"Batch {i + 1}")
#         print(f"Inputs Shape: {inputs.shape}")
#         print(f"Targets Shape: {targets.shape}")
#         print(f"Inputs Sample: {inputs[0]}")  # 打印第一个样本的输入
#         print(f"Targets Sample: {targets[0]}")  # 打印第一个样本的目标
#
# # 检查训练数据加载器的输出
# check_dataloader_output(train_loader, num_batches_to_check=16)

# 检查 DataLoader 输出
# 检查 DataLoader 输出
# def check_dataloader_output(data_loader, num_batches_to_check=1):
#     for i, (inputs, targets) in enumerate(data_loader):
#         if i >= num_batches_to_check:
#             break
#         print(f"Batch {i + 1}")
#         print(f"Inputs Shape: {inputs.shape}")
#         print(f"Targets Shape: {targets.shape}")
#         print("Inputs Data:", inputs)  # 打印整个批次的输入数据
#         print("Targets Data:", targets)  # 打印整个批次的目标数据
#
#
# # 检查训练数据加载器的输出
# check_dataloader_output(train_loader, num_batches_to_check=1)
# class GRUModel(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
#         super(GRUModel, self).__init__()
#         self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
#
#
#     def forward(self, x):
#         output, _ = self.gru(x)
#
#         # 取最后一个时间步的输出
#         # output 的形状是 [batch_size, seq_len, hidden_dim]
#         # 取 seq_len 维度的最后一个元素
#         last_step_output = output[:, -1, :]
#         return last_step_output

# SEED = random.randint(0, 1000000)  # 生成一个随机的种子
# print("gurmodel Seed:", SEED)
# torch.cuda.manual_seed_all(SEED)


# class GRUModel(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_layers):
#         super(GRUModel, self).__init__()
#         self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
#
#     def forward(self, x):
#         output, _ = self.gru(x)
#
#         # 取最后一个时间步的输出
#         # output 的形状是 [batch_size, seq_len, hidden_dim]
#         # 我们取 seq_len 维度的最后一个元素
#         last_step_output = output[:, -1, :]
#         return last_step_output
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        # 添加一个全连接层来生成最终的输出
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        output, _ = self.gru(x)
        # 取最后一个时间步的输出
        last_step_output = output[:, -1, :]
        # 通过全连接层生成最终的输出
        predicted_output = self.fc(last_step_output)
        return predicted_output

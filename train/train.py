#coding: utf-8
import os
import time
import torch
os.environ['CUDA_LAUNCH_BLOCKING'] = "3"
import random
import torch.nn as nn
from sklearn.metrics import r2_score, mean_squared_error
# /home/yfj/anaconda3/envs/stock/bin/python3.9 train/Prediction.py
from HGTAN.hyperedge_attn import hyperedge_atten
from data_loader import train_loader
from data_loader import test_loader
from HGTAN.grumodel import GRUModel
from load_data import hyperedge_index
from HGTAN.HGNN import HGNN
from HGTAN import hyperedge_attn
from load_H import H
# 设置CPU的随机种子
# torch.manual_seed(SEED)
# random.seed(SEED)
# np.random.seed(SEED)
#
# # 检查是否有GPU可用，如果有，设置GPU的随机种子
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(SEED)
#     torch.cuda.manual_seed_all(SEED)  # 对于单个GPU，这行与torch.cuda.manual_seed效果相同
#     torch.backends.cudnn.deterministic = True  # 保证每次返回的卷积算法将是确定的，如果不设置，可能会因为卷积实现的随机性而导致不一致的结果
#     torch.backends.cudnn.benchmark = False  # 对于固定输入大小，开启此项可以提高运行效率，但对于不同输入大小，可能会导致结果不一致
SEED = random.randint(0, 1000000)  # 生成一个随机的种子
print("Random Seed:", SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# os.environ['CUDA_LAUNCH_BLOCKING'] = "3"
# 定义模型
gru_model = GRUModel(input_dim=1, hidden_dim=64, output_dim=1, num_layers=10).to(device)
# lstm_model = LSTMModel(input_dim=1, hidden_dim=10, output_dim=1, num_layers=1).to(device)
# hypergraph_model = HypergraphAttentionNetwork(in_channels=1, out_channels=1, attn_out_sz=32).to(device)
# hgnn_model = HGNN(in_channels=32, n_hid=32, dropout=0).to(device)
hyat_model = hyperedge_atten(nfeat=1, nhid=2000, dropout=0).to(device)
# gnn_model = GNNModel(num_features=1, hidden_dim=1, output_dim=1).to(device)
# gat_model = GAT(num_features=64, hidden_channels=4).to(device)
# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(list(gru_model.parameters()) + list(hyat_model.parameters()), lr=0.0001)
# optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.00001)

num_epochs = 400  # 设置训练的迭代次数
for epoch in range(num_epochs):
    epoch_start_time = time.time()  # 开始记录时间
    gru_model.train()  # 将GRU模型设置为训练模式
    hyat_model.train()  # 将超图神经网络模型设置为训练模式
    total_loss = 0
    total_batches = 0  # 初始化批次计数器
    for batch in train_loader:
        start_time = time.time()  # 记录训练开始时间
        # 清空过往梯度
        optimizer.zero_grad()
        gru_inputs, targets = batch[0].to(device), batch[1].to(device)
        # print(gru_inputs.shape,targets.shape)
        # 每一个batch中包含一个[0]为特征，[1]为目标数据。
        gru_outputs = gru_model(gru_inputs)
        # print(gru_outputs.shape)
        # gru_outputs = gru_outputs.to(device)
        # print(gru_outputs)
        # hyperedge_index = hyperedge_index.to(device)
        # edge_index = edge_index.to(device)
        H = H.to(device)
        predicted_output = hyat_model(gru_outputs, H).to(device)
        # print(predicted_output.shape,H.shape)
        # 超图神经网络输入
        # hypergraph_inputs = gru_outputs.to(device)
        # hyperedge_index = hyperedge_index.to(device)
        # hypergraph_outputs = hypergraph_model(hypergraph_inputs, hyperedge_index).to(device)
        # 计算损失
        loss = criterion(predicted_output, targets)
        # # 清空过往梯度
        # optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 优化
        optimizer.step()
        # 累计损失
        total_loss += loss.item()
        # total_batches += 1  # 更新批次计数器
        # print(
        #     f"Epoch {epoch + 1}, Batch {total_batches}, Loss: {loss.item()}, Time: {time.time() - start_time} seconds")
    epoch_end_time = time.time()  # 结束记录时间
    epoch_duration = epoch_end_time - epoch_start_time
    epoch_avg_loss = total_loss / len(train_loader)
    # 打印损失
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_avg_loss:.8f}, Duration: {epoch_duration:.2f}')

    # gru_model.eval()
    # gat_model.eval()
    # val_loss = 0
    # with torch.no_grad():
    #     for val_batch in val_loader:
    #         val_inputs, val_targets = val_batch[0].to(device), val_batch[1].to(device)
    #         val_gru_outputs = gru_model(val_inputs)
    #         val_gru_outputs = val_gru_outputs.to(device)
    #         edge_index = edge_index.to(device)
    #         val_predicted_output = gat_model(val_gru_outputs, edge_index).to(device)
    #         # val_hypergraph_inputs = val_gru_outputs.to(device)
    #         # val_hypergraph_outputs = hypergraph_model(val_hypergraph_inputs, hyperedge_index).to(device)
    #         val_batch_loss = criterion(val_predicted_output, val_targets)
    #         val_loss += val_batch_loss.item()
    #
    # val_avg_loss = val_loss / len(val_loader)
    # print(f'Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_avg_loss:.8f}')

    if (epoch + 1) % 1 == 0: # if (epoch + 1) % 10 == 0:
        gru_model.eval()
        hyat_model.eval()
        all_targets = []
        all_predictions = []
        with torch.no_grad():
            for test_batch in test_loader:
                test_inputs, test_targets = test_batch[0].to(device), test_batch[1].to(device)
                test_gru_outputs = gru_model(test_inputs)
                test_gru_outputs = test_gru_outputs.to(device)
                # hyperedge_index = hyperedge_index.to(device)
                # hyperedge_index = hyperedge_index.to(device)
                # edge_index = edge_index.to(device)
                H = H.to(device)
                test_predicted_output = hyat_model(test_gru_outputs, H).to(device)
                # test_hypergraph_inputs = test_gru_outputs.to(device)
                # test_hypergraph_outputs = hypergraph_model(test_hypergraph_inputs, hyperedge_index).to(device)
                # 收集所有的预测和目标
                all_predictions.extend(test_predicted_output.cpu().numpy())
                all_targets.extend(test_targets.cpu().numpy())

        # 计算 R2 分数
        r2 = r2_score(all_targets, all_predictions)
        # absolute_errors = torch.abs(all_predictions - all_targets)
        # mae = torch.mean(absolute_errors)
        mse = mean_squared_error(all_targets, all_predictions)
        print(f'Epoch {epoch + 1}/{num_epochs}, Test R2 Score: {r2:.8f}, Test MSE Score: {mse:.8f}')

save_directory = "../prediction"
torch.save(gru_model, os.path.join(save_directory, 'gru_model_complete_400_log_test.pth'))
torch.save(hyat_model, os.path.join(save_directory, 'hypergraph_model_complete_400_log_test.pth'))
print("Models saved in 'models/' directory.")

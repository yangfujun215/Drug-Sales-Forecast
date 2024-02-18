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

SEED = random.randint(0, 1000000)  # 生成一个随机的种子
print("Random Seed:", SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义模型
gru_model = GRUModel(input_dim=1, hidden_dim=32, output_dim=1, num_layers=10).to(device)

# hyat_model = hyperedge_atten(nfeat=1, nhid=2000, dropout=0).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(list(gru_model.parameters()), lr=0.0001)


num_epochs = 400  # 设置训练的迭代次数
for epoch in range(num_epochs):
    epoch_start_time = time.time()  # 开始记录时间
    gru_model.train()  # 将GRU模型设置为训练模式

    total_loss = 0
    total_batches = 0  # 初始化批次计数器
    for batch in train_loader:
        start_time = time.time()  # 记录训练开始时间
        # 清空过往梯度
        optimizer.zero_grad()
        gru_inputs, targets = batch[0].to(device), batch[1].to(device)
        # 每一个batch中包含一个[0]为特征，[1]为目标数据。
        gru_outputs = gru_model(gru_inputs)


        # 计算损失
        loss = criterion(gru_outputs, targets)
        # # 清空过往梯度
        # optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 优化
        optimizer.step()
        # 累计损失
        total_loss += loss.item()
    epoch_end_time = time.time()  # 结束记录时间
    epoch_duration = epoch_end_time - epoch_start_time
    epoch_avg_loss = total_loss / len(train_loader)
    # 打印损失
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_avg_loss:.8f}, Duration: {epoch_duration:.2f}')
    if (epoch + 1) % 1 == 0: # if (epoch + 1) % 10 == 0:
        gru_model.eval()

        all_targets = []
        all_predictions = []
        with torch.no_grad():
            for test_batch in test_loader:
                test_inputs, test_targets = test_batch[0].to(device), test_batch[1].to(device)
                test_gru_outputs = gru_model(test_inputs)
                test_gru_outputs = test_gru_outputs.to(device)

                test_predicted_output = test_gru_outputs
                # test_hypergraph_inputs = test_gru_outputs.to(device)
                # test_hypergraph_outputs = hypergraph_model(test_hypergraph_inputs, hyperedge_index).to(device)
                # 收集所有的预测和目标
                all_predictions.extend(test_predicted_output.cpu().numpy())
                all_targets.extend(test_targets.cpu().numpy())

        # 计算 R2 分数
        r2 = r2_score(all_targets, all_predictions)
        mse = mean_squared_error(all_targets, all_predictions)
        print(f'Epoch {epoch + 1}/{num_epochs}, Test R2 Score: {r2:.8f}, Test MSE Score: {mse:.8f}')

# save_directory = "../prediction"
# torch.save(gru_model, os.path.join(save_directory, 'gru_model_complete_400_log_test.pth'))
# torch.save(hyat_model, os.path.join(save_directory, 'hypergraph_model_complete_400_log_test.pth'))
print("Models saved in 'models/' directory.")

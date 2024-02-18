import os
import numpy as np
import torch
import pandas as pd
from data_loader import test_loader, drug_id
# from load_data import hyperedge_index
from load_H import H as hyperedge_index
from data_loader import scalers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

gru_model_path = '../prediction/gru_model_complete_400_log_test.pth'
hypergraph_model_path = '../prediction/hypergraph_model_complete_400_log_test.pth'
print("123")
if os.path.exists(gru_model_path) and os.path.exists(hypergraph_model_path):
    gru_model = torch.load(gru_model_path).to(device)
    hypergraph_model = torch.load(hypergraph_model_path).to(device)
    print("Models loaded successfully.")
else:
    print("Model files not found.")
    exit()
print(gru_model)
print(hypergraph_model)
# gru_model = GRUModel(input_dim=1, hidden_dim=100, output_dim=1, num_layers=1).to(device)
# hypergraph_model = HypergraphAttentionNetwork(in_channels=1, out_channels=1, attn_out_sz=32).to(device)

# gru_model.eval()  # 设置GRU模型为评估模式
# hypergraph_model.eval()  # 设置超图神经网络为评估模式
# print("123")
last_batch_inputs, last_batch_targets = None, None
for inputs, targets in test_loader:
    last_batch_inputs, last_batch_targets = inputs, targets

# 检查是否成功提取
if last_batch_inputs is not None and last_batch_targets is not None:
    print("Successfully extracted the last batch.")
else:
    print("Failed to extract the last batch.")

# 确保数据在正确的设备上
last_batch_inputs = last_batch_inputs.to(device)

# 不计算梯度
with torch.no_grad():
    gru_outputs = gru_model(last_batch_inputs)
    # 超图神经网络输入
    hypergraph_inputs = gru_outputs.to(device)
    hyperedge_index = hyperedge_index.to(device)
    print(hypergraph_inputs.shape,hyperedge_index.shape)
    hypergraph_outputs = hypergraph_model(hypergraph_inputs, hyperedge_index).to(device)
    predictions = hypergraph_outputs

# scaler = MinMaxScaler()
# predictions = scaler.inverse_transform(predictions.cpu().numpy())
predictions = predictions.cpu()
scaler = scalers[drug_id]  # 获取对应药品的归一化器
original_scale_predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
final_predictions = np.exp(original_scale_predictions) - 1
# 创建一个 DataFrame
predicted_df = pd.DataFrame(final_predictions, columns=['Predicted Sales'])
# 保存为 Excel 文件
predicted_df.to_csv("../prediction/predicted_sales2.csv", index=False)
# # 1. 获取对应药品的归一化器
# scaler = scalers[drug_id]
#
# # 2. 反归一化预测结果
# inverse_scaled_predictions = scaler.inverse_transform(final_predictions.reshape(-1, 1))
#
# # 3. 创建一个包含反归一化结果的 DataFrame
# predicted_df['Inverse_Scaled_Predictions'] = inverse_scaled_predictions
#
# # 4. 将包含反归一化结果的 DataFrame 保存为 Excel 文件
# predicted_df.to_excel("../prediction/predicted_sales_with_inverse_scaling.xlsx", index=False)

# # 如果您有药品的标识符或名称，也可以将它们添加到 DataFrame
# # predicted_df['Drug_ID'] = drug_ids  # 假设 drug_ids 包含药品的标识符或名称


print(predictions.shape)
print("Predicted Sales for the next day:", predictions)

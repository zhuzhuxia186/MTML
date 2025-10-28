from utils.select_data import create_dataloader
from pytorch_metric_learning import losses,reducers,miners,distances
from sklearn.metrics import confusion_matrix
import torch
import torch.nn  as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
from openpyxl import Workbook
from sklearn.metrics import f1_score, cohen_kappa_score  # 引入F1分数和Kappa系数
from conformer import *
import os

def lrSwap22c_left_right(data_x,data_y):
    data_x = data_x.cpu().numpy()  # Move to CPU and convert to NumPy
    data_y = data_y.cpu().numpy()  # Move to CPU and convert to NumPy
    tempX = data_x.copy()
    tempy = data_y.copy()
    mirrorMap = [1, 6, 5, 4, 3, 2, 13, 12, 11, 10, 9, 8, 7, 18, 17, 16, 15, 14, 21, 20, 19, 22]
    mirrorMap = np.array(mirrorMap) - 1
    for i in range(22):
        tempX[:, :,i ,:] = data_x[:, :,mirrorMap[i],:]
    for i in range(data_x.shape[0]):
        if data_y[i] % 10 == 0:
            tempy[i] = int(data_y[i] + 1)
        elif data_y[i] % 10 == 1:
            tempy[i] = int(data_y[i] - 1)

    tempX = torch.from_numpy(tempX).to('cuda')
    tempy = torch.from_numpy(tempy).to('cuda')
    return tempX,tempy



def train_generic(low,sub_id,dataset_name, modelName, lr, epochs, batch_size, save_path, n_class, n_ch, sample_num,
                  criterion_another=None, criterion_another_weight=None, device='cuda', log_interval=1):

    # 加载数据
    train_loader, test_loader = create_dataloader(batch_size, dataset_name,sub_id,low)

    # 初始化模型
    model = Conformer()
    model = model.to(device)
    # 初始化优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(b1, b2))
    # 初始化优化器
    criterion = torch.nn.CrossEntropyLoss().cuda()

    # 初始化镜像对比代价
    distance = distances.CosineSimilarity()
    triple_loss = losses.TripletMarginLoss(margin=0.05, distance=distance, reducer=reducers.AvgNonZeroReducer())
    #这里的TripletMarginMiner里面的全部文件要替换为utils下的triplet_margin_miner.py
    mining_func = miners.TripletMarginMiner(
        margin=0.05, distance=distance, type_of_triplets="semihard"
    )
    # 创建工作簿
    wb = Workbook()
    ws = wb.active
    columns = ['epoch', 'train_loss', 'train_acc',  'test_loss', 'test_acc']
    ws.append(columns)
    # 训练循环
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (data, label) in pbar:
            data, label = data.to(device), label.to(device)
            mirror_data, mirror_label = lrSwap22c_left_right(data, label)
            mirror_data, mirror_label = mirror_data.to(device), mirror_label.to(device)
            optimizer.zero_grad()
            ori_feature,output = model(data)
            mirror_fc, mirror_output = model(mirror_data)
            indices_tuple = mining_func(ori_feature, label, mirror_fc, mirror_label)
            triple_loss_output = triple_loss(ori_feature, label, indices_tuple, mirror_fc, mirror_label)
            # 合并输出
            combined_outputs = torch.cat((output, mirror_output), dim=0)
            # 合并标签
            combined_labels = torch.cat((label, mirror_label), dim=0)
            criterion_loss = criterion(combined_outputs, combined_labels)
            loss = criterion_loss + triple_loss_output
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            train_acc += pred.eq(label.view_as(pred)).sum().item()

            if i % log_interval == 0:
                pbar.set_description(
                    f'Epoch: {epoch}, Loss: {train_loss / (i + 1):.4f}, Acc: {train_acc /   len(train_loader.dataset):.4f}')
        model.eval()
        # 测试阶段
        test_loss = 0.0
        test_acc = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for data, label in test_loader:
                data, label = data.to(device), label.to(device)
                ori_feature, output = model(data)
                test_loss += criterion(output, label).item()
                pred = output.argmax(dim=1, keepdim=True)
                test_acc += pred.eq(label.view_as(pred)).sum().item()
                # 收集所有预测和真实标签
                all_preds.extend(pred.cpu().numpy().flatten())
                all_labels.extend(label.cpu().numpy().flatten())


        test_loss /= len(test_loader.dataset)
        test_acc /= len(test_loader.dataset)


        # 保存当前epoch的模型参数
        torch.save(model.state_dict(),
                   rf'/2a_save_model_martix/jiaocha/model\sub{sub_id}\{sub_id}_epoch_{epoch}_weights.pth')
        # 记录日志
        ws.append([epoch, train_loss / len(train_loader.dataset), train_acc /  len(train_loader.dataset),
                   test_loss, test_acc])

    wb.save(f'{save_path}_log.xlsx')


dataset_name=['2a']
model_name_list = ['conformer']
n_class = 4
n_ch = 0
lr = 0.0002
b1 = 0.5
b2 = 0.999
epochs = 500
batch_size = 48
sample_input = 1125
sub_list=[1,2,3,4,5,6,7,8,9]
low_hz=[0]
for dataName in dataset_name:
    if dataName == '2a':
        n_class = 4
        n_ch = 22
    for model_name in model_name_list:
        for lun in range(1,2):
            for sub_id in sub_list:
                for low in low_hz:
                    save_file_path = rf'D:\eeg_conformer\2a_save_model_martix\acc\{lun}\sub{sub_id}_{model_name}_{low}hz'
                    train_generic(low, sub_id, dataName, model_name, lr, epochs, batch_size, save_file_path, n_class,
                                  n_ch, sample_input,
                                  criterion_another=None,
                                  criterion_another_weight=None, device='cuda', log_interval=1)







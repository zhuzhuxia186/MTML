from utils.process_2a import read_shu_2a
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
def select_dataset(dataset_type,sub_id,low):
    if dataset_type == '2a':
        data_path_2a = r'D:\BCI\BCICIV_2a_gdf'
        return read_shu_2a((low,38),data_path_2a,sub_id)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def create_dataloader(batch_size,dataset_type,sub_id,low):
    train_set,test_set=select_dataset(dataset_type,sub_id,low)
    train_data=train_set.X
    train_label=train_set.y

    test_data = test_set.X
    test_data_label = test_set.y


    train_data, train_label = torch.from_numpy(train_data), torch.from_numpy(train_label)
    #valid_data, valid_data_label = torch.from_numpy(valid_data), torch.from_numpy(valid_data_label)
    test_data, test_data_label = torch.from_numpy(test_data), torch.from_numpy(test_data_label)
    dataset = TensorDataset(train_data, train_label)
    dataloader_train = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # data_set_valid = TensorDataset(valid_data, valid_data_label)
    # dataloader_valid = DataLoader(data_set_valid, batch_size=batch_size, shuffle=True)
    data_set_test = TensorDataset(test_data, test_data_label)
    dataloader_test = DataLoader(data_set_test, batch_size=batch_size, shuffle=True)

    return dataloader_train,  dataloader_test
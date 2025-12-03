import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def data_loader(Xs, labels, batch_size, num_workers=0, shuffle=True):
    # 轉換為 Tensor
    data_tensor = torch.tensor(Xs, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    # 創建 DataLoader
    dataset = TensorDataset(data_tensor, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    return dataloader


class Sampler:
    def __init__(self, X_train, y_train, n_classes, n_samples_per_class):
        self.X_train = X_train
        self.y_train = y_train
        self.n_classes = n_classes
        self.n_samples_per_class = n_samples_per_class
        self.batch_size = n_classes * n_samples_per_class
        
    def sample_epoch(self, batch_shuffle=True):
        """均衡取樣，每個 batch 內每個類別取 n_samples_per_class 個樣本"""
        label_index = [np.where(self.y_train == i)[0] for i in range(self.n_classes)]
        
        max_class_idx = np.argmax([len(label_index[i]) for i in range(self.n_classes)])
        end_sample = False

        X_batches, y_batches = [], []
        
        while not end_sample:
            if len(label_index[max_class_idx]) < self.n_samples_per_class:
                end_sample = True

            batch_index_all = []
            for i in range(self.n_classes):
                if len(label_index[i]) < self.n_samples_per_class:
                    batch_index = label_index[i]
                    label_index[i] = np.where(self.y_train == i)[0]
                    
                    need_samples = self.n_samples_per_class - len(batch_index)
                    batch_index_2 = np.random.choice(label_index[i], need_samples, replace=len(label_index[i]) < need_samples)
                    
                    batch_index = np.concatenate([batch_index, batch_index_2])
                    label_index[i] = np.setdiff1d(label_index[i], batch_index_2)
                else:
                    batch_index = np.random.choice(label_index[i], self.n_samples_per_class, replace=False)
                    label_index[i] = np.setdiff1d(label_index[i], batch_index)
                    
                batch_index_all.append(batch_index)

            batch_index_all = np.concatenate(batch_index_all)
            if batch_shuffle:
                np.random.shuffle(batch_index_all)

            X_batches.append(self.X_train[batch_index_all])
            y_batches.append(self.y_train[batch_index_all])
            
        
        return np.concatenate(X_batches, axis=0), np.concatenate(y_batches, axis=0)

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

def read_file(filename):
	# Separator is automatically found
	f = pd.read_csv(filename, header=None, engine='python', sep=None)
	# print(f)
	inputs = f.iloc[:, :-1]
	outputs = f.iloc[:, -1]

	# inputs = f.values[:,0:(-1)]
	# outputs = f.values[:,(-1)]
	return inputs, outputs

def load_dataset(dataset_path):
	try:
		# Creating dicts for all partitions (saving partition order as keys)
		partition_list = {filename[filename.find('.') + 1:]: {} for filename
																in os.listdir(dataset_path)
																if filename.startswith("train_")}
		# Loading each dataset
		for filename in os.listdir(dataset_path):
			if filename.startswith("train_"):
				train_inputs, train_outputs = read_file(os.path.join(dataset_path, filename))
				partition_list[filename[filename.find('.') + 1:]]["train_inputs"] = train_inputs
				partition_list[filename[filename.find('.') + 1:]]["train_outputs"] = train_outputs
			elif filename.startswith("test_"):
				test_inputs, test_outputs = read_file(os.path.join(dataset_path, filename))
				partition_list[filename[filename.find('.') + 1:]]["test_inputs"] = test_inputs
				partition_list[filename[filename.find('.') + 1:]]["test_outputs"] = test_outputs
	except OSError:
		raise ValueError("No such file or directory: '%s'" % dataset_path)
	except KeyError:
		raise RuntimeError("Found partition without train files: partition %s"
							% filename[filename.find('.') + 1:])
	# Saving partitions as a sorted list of (index, partition) tuples
	partition_list = sorted(partition_list.items(), key=(lambda t: int(t[0])))
	return partition_list
# --------------------------------------

def load_tabular_dataset(dataset_name):
    dataset = load_dataset(f'orca-python/orca_python/datasets/{dataset_name}')
    
    _, partition = dataset[0]

    X_train_full = partition['train_inputs']
    y_train_full = partition['train_outputs']
    X_test = partition['test_inputs']
    y_test = partition['test_outputs']

    # 合併所有資料
    X_all = pd.concat([X_train_full, X_test], axis=0)
    y_all = pd.concat([y_train_full, y_test], axis=0)
    
    for i in range(1, len(y_all.unique())+1):
        y_all = y_all.replace(i, i-1)
        
    X_all = np.array(X_all)
    y_all = np.array(y_all)
    
    # 隨機打亂
    indices = np.arange(len(X_all))
    np.random.seed(0)
    np.random.shuffle(indices)
    X_all = X_all[indices]
    y_all = y_all[indices]

    return X_all, y_all

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


import os
from src.utils import set_seed
import torch
from dlordinal.datasets import FGNet, Adience
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose
import numpy as np
import pandas as pd

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
    if dataset_name == 'abalone':
        dataset_path = f'datasets/datasets-orreview/discretized-regression/5bins/{dataset_name}/matlab'
    else:
        dataset_path = f'datasets/datasets-orreview/ordinal-regression/{dataset_name}/matlab'
    
    dataset = load_dataset(dataset_path)
    
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


def load_image_dataset(dataset_name, seed):
    if dataset_name.lower() == "fgnet":
        return get_fgnet_dataset(seed)
    elif dataset_name.lower() == "adience":
        return get_adience_dataset(seed)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def get_fgnet_dataset(seed):
    print("Loading FGNet dataset...")
    set_seed(seed)

    # Download the FGNet dataset
    fgnet_train = FGNet(
        root="./datasets",
        train=True,
        target_transform=np.array,
        transform=Compose([ToTensor()]),
    )
    fgnet_test = FGNet(
        root="./datasets",
        train=False,
        target_transform=np.array,
        transform=Compose([ToTensor()]),
    )

    batch_size = 32
    train_loader = DataLoader(fgnet_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(fgnet_test, batch_size=batch_size, shuffle=False)
    
    X_train_full = torch.Tensor()
    y_train_full = torch.Tensor()
    for batch_images, batch_labels in train_loader:
        X_train_full = torch.cat((X_train_full, batch_images), 0)
        y_train_full = torch.cat((y_train_full, batch_labels), 0).long()

    X_train_full = np.array(X_train_full)
    y_train_full = np.array(y_train_full)

    X_test = torch.Tensor()
    y_test = torch.Tensor()
    for batch_images, batch_labels in test_loader:
        X_test = torch.cat((X_test, batch_images), 0)
        y_test = torch.cat((y_test, batch_labels), 0).long()
        
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    return X_train_full, y_train_full, X_test, y_test

    
def get_adience_dataset(seed):
    print("Loading Adience dataset...")
    set_seed(seed)

    adience_train = Adience(
        root="./datasets/",
        train=True,
        test_size=0.8,
        transform=ToTensor(),
    )

    adience_test = Adience(
        root="./datasets/",
        train=False,
        test_size=0.8,
        transform=ToTensor(),
    )
    
    batch_size = 128
    train_loader = DataLoader(adience_train, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(adience_test, batch_size=batch_size, shuffle=False, num_workers=0)
        
    X_train_full = torch.Tensor()
    y_train_full = torch.Tensor()
    
    
    for batch_images, batch_labels in train_loader:
        X_train_full = torch.cat((X_train_full, batch_images), 0)
        y_train_full = torch.cat((y_train_full, batch_labels), 0).long()

    X_train_full = np.array(X_train_full)
    y_train_full = np.array(y_train_full)

    X_test = torch.Tensor()
    y_test = torch.Tensor()
    for batch_images, batch_labels in test_loader:
        X_test = torch.cat((X_test, batch_images), 0)
        y_test = torch.cat((y_test, batch_labels), 0).long()
        
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    return X_train_full, y_train_full, X_test, y_test

from src.utils import set_seed
import torch
from dlordinal.datasets import FGNet, Adience
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose
import numpy as np

def load_image_dataset(dataset_name, seed):
    if dataset_name.lower() == "fgnet":
        return get_fgnet_dataset(seed)
    elif dataset_name.lower() == "adience":
        return get_adience_dataset(seed)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def get_fgnet_dataset(seed):
    
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
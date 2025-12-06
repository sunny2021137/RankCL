import yaml
import torch
import numpy as np
import random

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False  # 禁用動態算法選擇
        torch.backends.cudnn.deterministic = True  # 強制使用確定性算法
        
    torch.use_deterministic_algorithms(True, warn_only=True)  # 確保 PyTorch 其他運算也使用確定性算法
        
def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
    
    
def sample_according_to_distribution(X_all, y_all, desired_distribution, most_samples=2000, random_state=42):
    """
    根據給定的類別比例與總樣本數，從平衡資料中降採樣生成不平衡版本。

    參數：
    - X_all: 原始特徵資料（np.ndarray）
    - y_all: 原始標籤資料（np.ndarray）
    - desired_distribution: dict，如 {0:0.4, 1:0.3, ...}
    - total_samples: 目標總樣本數
    - random_state: 重現性

    回傳：
    - X_sampled, y_sampled: 採樣後的不平衡資料
    """
    np.random.seed(random_state)
    selected_indices = []

    for label, ratio in desired_distribution.items():
        label_indices = np.where(y_all == label)[0]
        target_count = int(ratio * most_samples)
        
        
        sampled = np.random.choice(label_indices, size=target_count, replace=False)
        selected_indices.extend(sampled)

    # 打亂結果
    selected_indices = np.array(selected_indices)
    np.random.shuffle(selected_indices)

    return X_all[selected_indices], y_all[selected_indices]


def make_distributions(distribution, n_classes, X_all, y_all, seed):
    # 找出最少樣本數的類別數量
    min_bin_len = len(y_all[y_all == 0])
    for i in range(n_classes):
        if len(y_all[y_all == i])< min_bin_len:
            min_bin_len = len(y_all[y_all == i])
            
    if distribution == "CH7":
        desired_distribution = {0:0.143,1:0.571,2:1.000,3:0.571,4:0.143}
        X_all, y_all = sample_according_to_distribution(X_all, y_all, desired_distribution, most_samples=min_bin_len, random_state=seed)
        # print_label_distribution(y_all)
        
    elif distribution == "CH20":
        desired_distribution = {0:0.05,1:0.525,2:1.000,3:0.525,4:0.05}
        X_all, y_all = sample_according_to_distribution(X_all, y_all, desired_distribution, most_samples=min_bin_len, random_state=seed)
        # print_label_distribution(y_all)
    
    elif distribution == "D7":
        desired_distribution = {0:1.000,1:0.614,2:0.377,3:0.232,4:0.143}
        X_all, y_all = sample_according_to_distribution(X_all, y_all, desired_distribution, most_samples=min_bin_len, random_state=seed)
        # print_label_distribution(y_all)
    
    elif distribution == "D20":
        desired_distribution = {0:1.000,1:0.472,2:0.223,3:0.105,4:0.050}
        X_all, y_all = sample_according_to_distribution(X_all, y_all, desired_distribution, most_samples=min_bin_len, random_state=seed)
        # print_label_distribution(y_all)
    
    else:
        raise ValueError("Unsupported merge strategy")
    
    return X_all, y_all    
    
def print_label_distribution(y):
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    print("Class Distribution:")
    for label, count in zip(unique, counts):
        ratio = count / total
        print(f"  Class {label}: {count} samples ({ratio:.2%})")

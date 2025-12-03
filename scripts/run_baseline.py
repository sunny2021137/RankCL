import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from src.eval import evaluate_metrics
from src.factory import get_model_ml
import argparse
from src.utils import make_distributions, print_label_distribution, set_seed, load_yaml
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
import torch
from data.dataset import load_image_dataset, load_tabular_dataset
import gc
import torchvision.models as models
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

def get_pretrained_res18():
    res = models.resnet18(weights="IMAGENET1K_V1")
    res.fc = nn.Identity() # 去除最後分類層
    return res


def get_features(model, data_np, batch_size=64):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    features = []
    dataset = TensorDataset(torch.from_numpy(data_np))
    loader = DataLoader(dataset, batch_size=batch_size)
    
    with torch.no_grad():
        for batch in loader:
            batch_x = batch[0].to(device)
            output = model(batch_x).squeeze()
            features.append(output.cpu())
    
    return torch.cat(features).numpy()

def run_tabular_baseline(base_cfg):
    dataset_name = base_cfg["dataset"]["dataset_name"]
    
    X_all, y_all = load_tabular_dataset(dataset_name)
    n_classes = len(np.unique(y_all))
    
    # undersample according to distribution if specified
    distribution = base_cfg["dataset"].get("distribution", None)
    if distribution is not None:
        X_all, y_all = make_distributions(distribution, n_classes, X_all, y_all, base_cfg["seed"])
        print_label_distribution(y_all)
    
    # output directory
    if distribution is not None:
        out_dir = base_cfg["out_dir"] + f"/{base_cfg['dataset']['dataset_type']}/" + f"{dataset_name}_{distribution}" + "/" + base_cfg["method_name"] + "/"
    else:
        out_dir = base_cfg["out_dir"] + f"/{base_cfg['dataset']['dataset_type']}/" + dataset_name + "/" + base_cfg["method_name"] + "/"
    os.makedirs(out_dir, exist_ok=True)
    
    results = []   
    cm_norm_sum = 0
    run = 0
    skf_outer = StratifiedKFold(n_splits=base_cfg["train"]["n_runs"], shuffle=True, random_state=base_cfg["seed"])
    for train_idx_outer, test_idx_outer in skf_outer.split(X_all, y_all):

        seed = base_cfg["seed"] + run
        set_seed(seed)
        
        print(f"--------------Run {run} with seed {seed}--------------")

        # load best hyperparams
        if distribution is not None:
            seed_config_path = f"{base_cfg['best_hyperparams_dir']}/{base_cfg['dataset']['dataset_type']}/{dataset_name}_{distribution}/{base_cfg['method_name']}/seed{base_cfg['seed']}_run{run}.yaml"
        else:
            seed_config_path = f"{base_cfg['best_hyperparams_dir']}/{base_cfg['dataset']['dataset_type']}/{dataset_name}/{base_cfg['method_name']}/seed{base_cfg['seed']}_run{run}.yaml"
        if not os.path.exists(seed_config_path):
            raise FileNotFoundError(f"Best hyperparameters file not found: {seed_config_path}")
        hyper_cfg = load_yaml(seed_config_path)
        
        
        # 合併設定（base 為底，hyper 覆蓋）
        config = base_cfg | hyper_cfg
    
        X_train_full, X_test = X_all[train_idx_outer], X_all[test_idx_outer]
        y_train_full, y_test = y_all[train_idx_outer], y_all[test_idx_outer]
        
        X_train_run = X_train_full.astype(np.float32)
        X_test_run = X_test.astype(np.float32)
        
        # NOTE:
        if config["use_reweight"] == True:
            # not implemented for none-deep models
            raise NotImplementedError("Reweighting not implemented for none-deep models.")

        model = get_model_ml(config["base_method_name"], config["search_params"]) 
        model.fit(X_train_run, y_train_full)
        y_pred = model.predict(X_test_run)
        result = evaluate_metrics(y_test, y_pred, n_classes)
        results.append(result)
        
        cm = confusion_matrix(y_test, y_pred, labels=range(0, n_classes))
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm_sum += cm_norm
        
        run += 1

    cm_norm_avg = cm_norm_sum / config["train"]["n_runs"]
    cm = np.array(cm_norm_avg)  # 轉成純 numpy array
    np.save(f"{out_dir}/confusion_matrix.npy", cm)

        
    # 平均、標準差
    results_df = pd.DataFrame(results)
    mean_values = results_df.mean().rename(lambda x: f"{x}_mean")
    std_values = results_df.std().rename(lambda x: f"{x}_std")
    result_df = pd.DataFrame([pd.concat([mean_values, std_values])])
    # 存成 CSV 檔案
    result_df.to_csv(f"{out_dir}/metrics.csv", index=False)  # index=False 避免存入行索引
    print(f"CSV 檔案已存成 {out_dir}/metrics.csv")

def run_image_baseline(base_cfg):
    
    dataset_name = base_cfg["dataset"]["dataset_name"]
    seed = base_cfg["seed"]
    X_train_full, y_train_full, X_test, y_test = load_image_dataset(dataset_name, seed)
    n_classes = len(np.unique(y_train_full))
     
    # output directory
    out_dir = base_cfg["out_dir"] + f"/{base_cfg['dataset']['dataset_type']}/" + dataset_name + "/" + base_cfg["method_name"] + "/"
    os.makedirs(out_dir, exist_ok=True)
    
    # load best hyperparams
    seed_config_path = f"{base_cfg['best_hyperparams_dir']}/{base_cfg['dataset']['dataset_type']}/{dataset_name}/{base_cfg['method_name']}/seed{base_cfg['seed']}.yaml"
    if not os.path.exists(seed_config_path):
        raise FileNotFoundError(f"Best hyperparameters file not found: {seed_config_path}")
    hyper_cfg = load_yaml(seed_config_path)
    
    # 合併設定（base 為底，hyper 覆蓋）
    config = base_cfg | hyper_cfg
    
    results = []
    cm_norm_sum = 0
    for run in range(config["train"]["n_runs"]):
        seed = config['seed'] + run
        set_seed(seed)
                
        # 固定精度
        X_train_run = X_train_full.astype(np.float32)
        X_test_run = X_test.astype(np.float32)
                    
        if config["use_reweight"] == True:
            # not implemented for none-deep models
            raise NotImplementedError("Reweighting not implemented for none-deep models.")
                    
        res = get_pretrained_res18()
        feature_train = get_features(res, X_train_run)
        feature_test = get_features(res, X_test_run)
        
        model = get_model_ml(config["base_method_name"], config["search_params"])    
        model.fit(feature_train, y_train_full)
        y_pred = model.predict(feature_test)
        
        result = evaluate_metrics(y_test, y_pred, n_classes)
        results.append(result)
        
        cm = confusion_matrix(y_test, y_pred, labels=range(0, n_classes))
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm_sum += cm_norm
        
    cm_norm_avg = cm_norm_sum / config["train"]["n_runs"]
    cm = np.array(cm_norm_avg)  # 轉成純 numpy array
    np.save(f"{out_dir}/confusion_matrix.npy", cm)
    
    # 平均、標準差
    results_df = pd.DataFrame(results)
    mean_values = results_df.mean().rename(lambda x: f"{x}_mean")
    std_values = results_df.std().rename(lambda x: f"{x}_std")
    result_df = pd.DataFrame([pd.concat([mean_values, std_values])])
    # 存成 CSV 檔案
    result_df.to_csv(f"{out_dir}/metrics.csv", index=False)  # index=False 避免存入行索引
    print(f"CSV 檔案已存成 {out_dir}/metrics.csv") 
                           
    gc.collect()
    
    
def main():
    parser = argparse.ArgumentParser(description="RankCL Framework")
    parser.add_argument("--config", type=str, default="configs/default_baseline.yaml")
    args = parser.parse_args()

    base_cfg = load_yaml(args.config)
    
    if base_cfg["dataset"]["dataset_type"] == "tabular":
        run_tabular_baseline(base_cfg)
    elif base_cfg["dataset"]["dataset_type"] == "image":
        run_image_baseline(base_cfg)
    else:
        raise ValueError("Unsupported dataset type")

if __name__ == "__main__":
    main()

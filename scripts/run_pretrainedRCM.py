import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import argparse
from train.trainer import RankCLPretrainedTrainer, RankCLTrainer
from src.utils import make_distributions, print_label_distribution, set_seed, load_yaml
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
import pandas as pd
from data.data_loader import data_loader, load_tabular_dataset, Sampler
import torch
from data.dataset import load_image_dataset
import gc

def run_pretrainedRCM_tabular(base_cfg):
    dataset_name = base_cfg["dataset"]["dataset_name"]
    
    X_all, y_all = load_tabular_dataset(dataset_name)
    n_classes = len(np.unique(y_all))
    
    # undersample according to distribution if specified
    distribution = base_cfg["dataset"].get("distribution", None)
    if distribution is not None:
        X_all, y_all = make_distributions(distribution, n_classes, X_all, y_all, base_cfg["seed"])
        print_label_distribution(y_all)
    
    # NOTE: output directory
    if distribution is not None:
        out_dir = base_cfg["out_dir"] + f"/pretrained/{base_cfg['dataset']['dataset_type']}/" + f"{dataset_name}_{distribution}" + "/" + base_cfg["method_name"] + "/"
    else:
        out_dir = base_cfg["out_dir"] + f"/pretrained/{base_cfg['dataset']['dataset_type']}/" + dataset_name + "/" + base_cfg["method_name"] + "/"
    os.makedirs(out_dir, exist_ok=True)
    
    results = []
    run = 0
    skf_outer = StratifiedKFold(n_splits=base_cfg["train"]["n_runs"], shuffle=True, random_state=base_cfg["seed"])
    for train_idx_outer, test_idx_outer in skf_outer.split(X_all, y_all):

        seed = base_cfg["seed"] + run
        set_seed(seed)
        
        print(f"--------------Run {run} with seed {seed}--------------")

        # NOTE: load best hyperparams
        if distribution is not None:
            seed_config_path = f"/pretrained/{base_cfg['best_hyperparams_dir']}/{base_cfg['dataset']['dataset_type']}/{dataset_name}_{distribution}/{base_cfg['method_name']}/seed{base_cfg['seed']}_run{run}.yaml"
        else:
            seed_config_path = f"/pretrained/{base_cfg['best_hyperparams_dir']}/{base_cfg['dataset']['dataset_type']}/{dataset_name}/{base_cfg['method_name']}/seed{base_cfg['seed']}_run{run}.yaml"
        if not os.path.exists(seed_config_path):
            raise FileNotFoundError(f"Best hyperparameters file not found: {seed_config_path}")
        hyper_cfg = load_yaml(seed_config_path)
        
        
        # 合併設定（base 為底，hyper 覆蓋）
        config = base_cfg | hyper_cfg
    
        X_train_full, X_test = X_all[train_idx_outer], X_all[test_idx_outer]
        y_train_full, y_test = y_all[train_idx_outer], y_all[test_idx_outer]
        
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=base_cfg["train"]["val_ratio"], stratify=y_train_full, random_state=seed
        )
        X_train_run = X_train.astype(np.float32)
        X_val_run = X_val.astype(np.float32)
        X_test_run = X_test.astype(np.float32)

        # 調整 batch size 為類別數的倍數，以符合 RankCL balanced dataloader的需求
        config["train"]["batch_size"] = (config["train"]["batch_size"] // n_classes) * n_classes
        
        # NOTE:
        train_sampler = Sampler(X_train_run, y_train, n_classes, n_samples_per_class=config["train"]["batch_size"]//n_classes)
        val_sampler = Sampler(X_val_run, y_val, n_classes, n_samples_per_class=config["train"]["batch_size"]//n_classes)
        test_sampler = Sampler(X_test_run, y_test, n_classes, n_samples_per_class=config["train"]["batch_size"]//n_classes)
    
        x_dim = X_train_run.shape[1]
        
        # NOTE:
        trainer = RankCLPretrainedTrainer(config, train_sampler, val_sampler, test_sampler, n_classes, x_dim)
        # train
        trainer.train()
        # NOTE: test
        result = trainer.test_loss()
        # results.append(result)
        
        # TODO: save model
        torch.save(trainer.model.state_dict(), os.path.join(out_dir, f"model_seed{base_cfg['seed']}_run{run}.pth"))
        
        
        run += 1
        

# TODO: 
def run_pretrainedRCM_image(base_cfg):
    torch.set_num_threads(8)
    
    dataset_name = base_cfg["dataset"]["dataset_name"]
    seed = base_cfg["seed"]
    X_train_full, y_train_full, X_test, y_test = load_image_dataset(dataset_name, seed)
    n_classes = len(np.unique(y_train_full))
     
    # NOTE: output directory
    out_dir = base_cfg["out_dir"] + f"/pretrained/{base_cfg['dataset']['dataset_type']}/" + dataset_name + "/" + base_cfg["method_name"] + "/"
    os.makedirs(out_dir, exist_ok=True)
    
    # NOTE: load best hyperparams
    seed_config_path = f"pretrained/{base_cfg['best_hyperparams_dir']}/{base_cfg['dataset']['dataset_type']}/{dataset_name}/{base_cfg['method_name']}/seed{base_cfg['seed']}.yaml"
    if not os.path.exists(seed_config_path):
        raise FileNotFoundError(f"Best hyperparameters file not found: {seed_config_path}")
    hyper_cfg = load_yaml(seed_config_path)
    
    # 合併設定（base 為底，hyper 覆蓋）
    config = base_cfg | hyper_cfg
    config["train"]["batch_size"] = (config["train"]["batch_size"] // n_classes) * n_classes
   
    results = []
    for run in range(config["train"]["n_runs"]):
        seed = config['seed'] + run
        set_seed(seed)
        
        # 切validation set
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=config["train"]["val_ratio"], stratify=y_train_full, random_state=seed
        )
                
        # 固定精度
        X_train_run = X_train.astype(np.float32)
        X_val_run = X_val.astype(np.float32)
        X_test_run = X_test.astype(np.float32)
        
        # NOTE:   
        train_sampler = Sampler(X_train_run, y_train, n_classes, n_samples_per_class=config["train"]["batch_size"]//n_classes)
        val_sampler = Sampler(X_val_run, y_val, n_classes, n_samples_per_class=config["train"]["batch_size"]//n_classes)
        test_sampler = Sampler(X_test_run, y_test, n_classes, n_samples_per_class=config["train"]["batch_size"]//n_classes)

        x_dim = X_train_run.shape[1] 
        
        # NOTE:
        trainer = RankCLPretrainedTrainer(params, train_sampler, val_sampler, test_sampler, n_classes, x_dim)

        trainer = RankCLTrainer(config, train_sampler, train_loader, val_loader, test_loader, n_classes, x_dim)
        # train
        trainer.train()
        # NOTE: test
        result = trainer.test_loss()
        # results.append(result)
        
        # TODO: save model???要不要run?
        torch.save(trainer.model.state_dict(), os.path.join(out_dir, f"model_seed{base_cfg['seed']}_run{run}.pth"))
                           
    gc.collect()
    
    
def main():
    parser = argparse.ArgumentParser(description="RankCL Framework")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    base_cfg = load_yaml(args.config)
    
    if base_cfg["dataset"]["dataset_type"] == "tabular":
        run_tabular(base_cfg)
    elif base_cfg["dataset"]["dataset_type"] == "image":
        run_image(base_cfg)
    else:
        raise ValueError("Unsupported dataset type")

if __name__ == "__main__":
    main()

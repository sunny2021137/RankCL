from copy import deepcopy
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import argparse
from train.trainer import BaselineTrainer, RankCLPretrainedTrainer, RankCLTrainer
from src.utils import make_distributions, print_label_distribution, set_seed, load_yaml
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
import pandas as pd
from data.data_loader import data_loader, load_tabular_dataset, Sampler
import torch
from data.dataset import load_image_dataset
import gc

def run_ptrcm_ftrcmob_tabular(base_cfg):
    if base_cfg['base_method_name'] != 'OBDECOC':
        raise ValueError("Only OBDECOC is supported as base method for PTRCM + FTRCM-OBD.")
    
    # CHANGE:
    config_pretrain = base_cfg.pop("pretrain")
    config_finetune = base_cfg.pop("finetune")
    
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
        
        
        # CHANGE:
        params = deepcopy(base_cfg)
        params["train"]["batch_size"] = (params["train"]["batch_size"] // n_classes) * n_classes

        X_train_full, X_test = X_all[train_idx_outer], X_all[test_idx_outer]
        y_train_full, y_test = y_all[train_idx_outer], y_all[test_idx_outer]
        
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=params["train"]["val_ratio"], stratify=y_train_full, random_state=seed
        )
        X_train_run = X_train.astype(np.float32)
        X_val_run = X_val.astype(np.float32)
        X_test_run = X_test.astype(np.float32)

        # CHANGE:
        x_dim = X_train_run.shape[1]
        
        # CHANGE:
        print("params 1:", params)
        params = deepcopy(base_cfg | config_pretrain | hyper_cfg)
        params["train"]["batch_size"] = (params["train"]["batch_size"] // n_classes) * n_classes
        print("params 2:", params)
        
        # NOTE: Pre-training RCM
        train_sampler = Sampler(X_train_run, y_train, n_classes, n_samples_per_class=params["train"]["batch_size"]//n_classes)
        val_sampler = Sampler(X_val_run, y_val, n_classes, n_samples_per_class=params["train"]["batch_size"]//n_classes)
        test_sampler = Sampler(X_test_run, y_test, n_classes, n_samples_per_class=params["train"]["batch_size"]//n_classes)
    
        pre_trainer = RankCLPretrainedTrainer(params, train_sampler, val_sampler, test_sampler, n_classes, x_dim)
        # train
        pre_trainer.train()
        
        pretrained_model_dict = pre_trainer.model.state_dict()
        # print("pretrained_model_dict:")
        # for k, v in pretrained_model_dict.items():
        #     print(f"{k}")
        
        # CHANGE:
        params = deepcopy(base_cfg | config_finetune | hyper_cfg)
        params["train"]["batch_size"] = (params["train"]["batch_size"] // n_classes) * n_classes
        print("params 3:", params)
                
        # NOTE: Fine-tuning RCM-OB
        train_sampler = Sampler(X_train_run, y_train, n_classes, n_samples_per_class=params["train"]["batch_size"]//n_classes)
        train_loader = data_loader(X_train_run, y_train, batch_size=params["train"]["batch_size"], num_workers=params["num_workers"], shuffle=False)
        val_loader = data_loader(X_val_run, y_val, batch_size=params["train"]["batch_size"], num_workers=params["num_workers"], shuffle=False)
        test_loader = data_loader(X_test_run, y_test, batch_size=params["train"]["batch_size"], num_workers=params["num_workers"], shuffle=False)

        finetune_trainer = RankCLTrainer(params, train_sampler, train_loader, val_loader, test_loader, n_classes, x_dim)
        finetune_model_dict = finetune_trainer.model.state_dict()
        # print("finetune_model_dict:")
        # for k, v in finetune_model_dict.items():
        #     print(f"{k}")
        
        # NOTE: 取代 weights
        rename_pretrained_dict = {}
        for k, v in pretrained_model_dict.items():
            new_key = f'base_classifier.{k}'
            rename_pretrained_dict[new_key] = v    
        pretrained_model_dict = {k: v for k, v in rename_pretrained_dict.items() if k in finetune_model_dict}
        # print("left:")
        # for k, v in pretrained_model_dict.items():
        #     print(f"{k}")
        
        finetune_model_dict.update(pretrained_model_dict)
        finetune_trainer.model.load_state_dict(finetune_model_dict)
        
        # train
        finetune_trainer.train()
        # test
        result = finetune_trainer.test()
                
        cm_norm_sum += result['confusion_matrix']
        del result['confusion_matrix']
        results.append(result)
        run += 1


    cm_norm_avg = cm_norm_sum / base_cfg["train"]["n_runs"]
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

def run_ptrcm_ftrcmob_image(base_cfg):
    if base_cfg['base_method_name'] != 'OBDECOC':
        raise ValueError("Only OBDECOC is supported as base method for PTRCM + FTRCM-OBD.")

    # CHANGE:
    config_pretrain = base_cfg.pop("pretrain")
    config_finetune = base_cfg.pop("finetune")
    
    torch.set_num_threads(8)
    
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
    
    # CHANGE:
    params = deepcopy(base_cfg)
    params["train"]["batch_size"] = (params["train"]["batch_size"] // n_classes) * n_classes
       
    results = []
    cm_norm_sum = 0
    for run in range(params["train"]["n_runs"]):
        seed = params['seed'] + run
        set_seed(seed)
        
        # 切validation set
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=params["train"]["val_ratio"], stratify=y_train_full, random_state=seed
        )
                
        # 固定精度
        X_train_run = X_train.astype(np.float32)
        X_val_run = X_val.astype(np.float32)
        X_test_run = X_test.astype(np.float32)
       
        x_dim = X_train_run.shape[1]
        
        # CHANGE:
        # print("params 1:", params)
        params = deepcopy(base_cfg | config_pretrain | hyper_cfg)
        params["train"]["batch_size"] = (params["train"]["batch_size"] // n_classes) * n_classes
        # print("params 2:", params)
            
        train_sampler = Sampler(X_train_run, y_train, n_classes, n_samples_per_class=params["train"]["batch_size"]//n_classes)
        val_sampler = Sampler(X_val_run, y_val, n_classes, n_samples_per_class=params["train"]["batch_size"]//n_classes)
        test_sampler = Sampler(X_test_run, y_test, n_classes, n_samples_per_class=params["train"]["batch_size"]//n_classes)

        pre_trainer = RankCLPretrainedTrainer(params, train_sampler, val_sampler, test_sampler, n_classes, x_dim)
        # train
        pre_trainer.train()
        
        pretrained_model_dict = pre_trainer.model.state_dict()
        # print("pretrained_model_dict:")
        # for k, v in pretrained_model_dict.items():
        #     print(f"{k}")
        
        # CHANGE:
        params = deepcopy(base_cfg | config_finetune | hyper_cfg)
        params["train"]["batch_size"] = (params["train"]["batch_size"] // n_classes) * n_classes
        # print("params 3:", params)
            
        # Fine-tuning RCM-OB
        train_sampler = Sampler(X_train_run, y_train, n_classes, n_samples_per_class=params["train"]["batch_size"]//n_classes)
        train_loader = data_loader(X_train_run, y_train, batch_size=params["train"]["batch_size"], num_workers=params["num_workers"], shuffle=False)
        val_loader = data_loader(X_val_run, y_val, batch_size=params["train"]["batch_size"], num_workers=params["num_workers"], shuffle=False)
        test_loader = data_loader(X_test_run, y_test, batch_size=params["train"]["batch_size"], num_workers=params["num_workers"], shuffle=False)
        
        finetune_trainer = RankCLTrainer(params, train_sampler, train_loader, val_loader, test_loader, n_classes, x_dim)
        finetune_model_dict = finetune_trainer.model.state_dict()
        # print("finetune_model_dict:")
        # for k, v in finetune_model_dict.items():
        #     print(f"{k}")
        
        # NOTE: 取代 weights
        rename_pretrained_dict = {}
        for k, v in pretrained_model_dict.items():
            new_key = f'base_classifier.{k}'
            rename_pretrained_dict[new_key] = v    
        pretrained_model_dict = {k: v for k, v in rename_pretrained_dict.items() if k in finetune_model_dict}
        # print("left:")
        # for k, v in pretrained_model_dict.items():
        #     print(f"{k}")
            
            
        finetune_model_dict.update(pretrained_model_dict)
        finetune_trainer.model.load_state_dict(finetune_model_dict)
        
        # train
        finetune_trainer.train()
        # test
        result = finetune_trainer.test()
        
        cm_norm_sum += result['confusion_matrix']
        del result['confusion_matrix']
        results.append(result)
        
    cm_norm_avg = cm_norm_sum / base_cfg["train"]["n_runs"]
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
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    base_cfg = load_yaml(args.config)
    
    if base_cfg["dataset"]["dataset_type"] == "tabular":
        run_ptrcm_ftrcmob_tabular(base_cfg)
    elif base_cfg["dataset"]["dataset_type"] == "image":
        run_ptrcm_ftrcmob_image(base_cfg)
    else:
        raise ValueError("Unsupported dataset type")

if __name__ == "__main__":
    main()

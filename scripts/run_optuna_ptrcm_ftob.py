from copy import deepcopy
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import argparse
import yaml
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
import optuna
import gc
from data.data_loader import data_loader, Sampler
from train.trainer import BaselineTrainer, RankCLPretrainedTrainer, RankCLTrainer
from src.utils import load_yaml, make_distributions, print_label_distribution, set_seed
from src.factory import get_optuna_params, get_pretrain_optuna_params
from data.dataset import load_image_dataset, load_tabular_dataset
import torch

def run_ptrcm_ftob_optuna_tabular(config):
    if config['base_method_name'] != 'OBDECOC':
        raise ValueError("Only OBDECOC is supported as base method for PTRCM + FTRCM-OBD.")
    
    # CHANGE:
    config_pretrain = config.pop("pretrain")
    config_finetune = config.pop("finetune")
    
    dataset_name = config["dataset"]["dataset_name"]

    X_all, y_all = load_tabular_dataset(dataset_name)
    n_classes = len(np.unique(y_all))
    
    # undersample according to distribution if specified
    distribution = config["dataset"].get("distribution", None)
    if distribution is not None:
        X_all, y_all = make_distributions(distribution, n_classes, X_all, y_all, config["seed"])
        print_label_distribution(y_all)
    
    skf_outer = StratifiedKFold(n_splits=config["train"]["n_runs"], shuffle=True, random_state=config["seed"])      
    run = 0
    for train_idx_outer, test_idx_outer in skf_outer.split(X_all, y_all):
        seed = config["seed"] + run
        set_seed(seed)
        
        print(f"--------------Run {run} with seed {seed}--------------")
        
        X_train_full, _ = X_all[train_idx_outer], X_all[test_idx_outer]
        y_train_full, _ = y_all[train_idx_outer], y_all[test_idx_outer]     
        
        skf = StratifiedKFold(n_splits=config["optuna"]["n_folds"], shuffle=True, random_state=seed)
    
        def objective(trial):
            # CHANGE:
            opt_params_pretrain = {}
            opt_params_pretrain["search_params"] = get_pretrain_optuna_params(trial)
            opt_params_finetune = {}
            opt_params_finetune["search_params"] = get_optuna_params(config["base_method_name"], config_finetune.get("use_rankcl", True), trial)
            # CHANGE:
            params = deepcopy(config)
            params["train"]["batch_size"] = (params["train"]["batch_size"] // n_classes) * n_classes
            
            cv_scores = []
            for train_idx, val_idx in skf.split(X_train_full, y_train_full):
                
                X_train_cv, X_val_cv = X_train_full[train_idx], X_train_full[val_idx]
                y_train_cv, y_val_cv = y_train_full[train_idx], y_train_full[val_idx]
                
                # 切early stop用的validation set
                X_train_e, X_val_e, y_train_e, y_val_e = train_test_split(
                    X_train_cv, y_train_cv, test_size=params["train"]["val_ratio"], stratify=y_train_cv, random_state=seed
                )
                                
                # 固定精度
                X_train_e = X_train_e.astype(np.float32)
                X_val_e = X_val_e.astype(np.float32)
                X_val_cv = X_val_cv.astype(np.float32)
                
                # net
                x_dim = X_train_e.shape[1]
                
                # CHANGE:
                # print("params 1:", params)
                params = deepcopy(config | config_pretrain | opt_params_pretrain)
                params["train"]["batch_size"] = (params["train"]["batch_size"] // n_classes) * n_classes
                # print("params 2:", params)
                
                # NOTE: Pre-training RCM
                train_sampler = Sampler(X_train_e, y_train_e, n_classes, n_samples_per_class=params["train"]["batch_size"]//n_classes)
                val_sampler = Sampler(X_val_e, y_val_e, n_classes, n_samples_per_class=params["train"]["batch_size"]//n_classes)
                test_sampler = Sampler(X_val_cv, y_val_cv, n_classes, n_samples_per_class=params["train"]["batch_size"]//n_classes)
                
                pre_trainer = RankCLPretrainedTrainer(params, train_sampler, val_sampler, test_sampler, n_classes, x_dim)
                # train
                pre_trainer.train()
                
                pretrained_model_dict = pre_trainer.model.state_dict()
                # print("pretrained_model_dict:")
                # for k, v in pretrained_model_dict.items():
                #     print(f"{k}")
                    
                # CHANGE:
                params = deepcopy(config | config_finetune | opt_params_finetune)
                params["train"]["batch_size"] = (params["train"]["batch_size"] // n_classes) * n_classes
                # print("params 3:", params)
                
                # NOTE: Fine-tuning OB
                train_loader = data_loader(X_train_e, y_train_e, batch_size=params["train"]["batch_size"], num_workers=params["num_workers"], shuffle=True) 
                val_loader = data_loader(X_val_e, y_val_e, batch_size=params["train"]["batch_size"], num_workers=params["num_workers"], shuffle=False)
                test_loader = data_loader(X_val_cv, y_val_cv, batch_size=params["train"]["batch_size"], num_workers=params["num_workers"], shuffle=False)
                
                finetune_trainer = BaselineTrainer(params, train_loader, val_loader, test_loader, n_classes, x_dim)
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
                                
                cv_scores.append(result[config["train"]["best_metric_name"]])
                
            # 計算folds平均分數
            return np.mean(cv_scores)
        
        # NOTE:
        study = optuna.create_study(direction=config["optuna"]["best_direction"], sampler=optuna.samplers.TPESampler(seed=seed))  # 目標是最大化驗證準確率
        study.optimize(objective, n_trials=config["optuna"]["n_trials"])  # 執行 3 次試驗
        
        # 輸出最佳超參數
        print("Best hyperparameters:", study.best_params)
           
        # NOTE:
        if distribution is not None:
            save_path = f'{config["best_hyperparams_dir"]}/{config["dataset"]["dataset_type"]}/{dataset_name}_{distribution}/{config["method_name"]}/seed{config["seed"]}_run{run}.yaml'
        else:
            save_path = f'{config["best_hyperparams_dir"]}/{config["dataset"]["dataset_type"]}/{dataset_name}/{config["method_name"]}/seed{config["seed"]}_run{run}.yaml'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        best_params_dict = {
            "search_params": study.best_params,
            "best_value": study.best_value,
            "best_trial": study.best_trial.number,
        }

        with open(save_path, "w") as f:
            yaml.safe_dump(best_params_dict, f, sort_keys=False)

        run += 1
        gc.collect()  


def run_ptrcm_ftob_optuna_image(config):
    if config['base_method_name'] != 'OBDECOC':
        raise ValueError("Only OBDECOC is supported as base method for PTRCM + FTRCM-OBD.")

    # CHANGE:
    config_pretrain = config.pop("pretrain")
    config_finetune = config.pop("finetune")
    
    seed = config["seed"]
    set_seed(seed)
    
    dataset_name = config["dataset"]["dataset_name"]

    X_train_full, y_train_full, _, _ = load_image_dataset(dataset_name, seed)
    n_classes = len(np.unique(y_train_full))
    
    skf = StratifiedKFold(n_splits=config["optuna"]["n_folds"], shuffle=True, random_state=seed)
    
    def objective(trial):
        # CHANGE:
        opt_params_pretrain = {}
        opt_params_pretrain["search_params"] = get_pretrain_optuna_params(trial)
        opt_params_finetune = {}
        opt_params_finetune["search_params"] = get_optuna_params(config["base_method_name"], config_finetune.get("use_rankcl", True), trial)
        
        # CHANGE:
        params = deepcopy(config)
        params["train"]["batch_size"] = (params["train"]["batch_size"] // n_classes) * n_classes
    
        cv_scores = []
    
        for train_idx, val_idx in skf.split(X_train_full, y_train_full):
            X_train_cv, X_val_cv = X_train_full[train_idx], X_train_full[val_idx]
            y_train_cv, y_val_cv = y_train_full[train_idx], y_train_full[val_idx]
            
            # 切early stop用的validation set
            X_train_e, X_val_e, y_train_e, y_val_e = train_test_split(
                X_train_cv, y_train_cv, test_size=params["train"]["val_ratio"], stratify=y_train_cv, random_state=seed
            )
                            
            # 固定精度
            X_train_e = X_train_e.astype(np.float32)
            X_val_e = X_val_e.astype(np.float32)
            X_val_cv = X_val_cv.astype(np.float32)
            
            x_dim = X_train_e.shape[1]
            
            # CHANGE:
            # print("params 1:", params)
            params = deepcopy(config | config_pretrain | opt_params_pretrain)
            params["train"]["batch_size"] = (params["train"]["batch_size"] // n_classes) * n_classes
            # print("params 2:", params)            
            
            # NOTE:
            train_sampler = Sampler(X_train_e, y_train_e, n_classes, n_samples_per_class=params["train"]["batch_size"]//n_classes)
            val_sampler = Sampler(X_val_e, y_val_e, n_classes, n_samples_per_class=params["train"]["batch_size"]//n_classes)
            test_sampler = Sampler(X_val_cv, y_val_cv, n_classes, n_samples_per_class=params["train"]["batch_size"]//n_classes)
            
            pre_trainer = RankCLPretrainedTrainer(params, train_sampler, val_sampler, test_sampler, n_classes, x_dim)
            # train
            pre_trainer.train()
            pretrained_model_dict = pre_trainer.model.state_dict()
            # print("pretrained_model_dict:")
            # for k, v in pretrained_model_dict.items():
            #     print(f"{k}")
            
            # CHANGE:
            params = deepcopy(config | config_finetune | opt_params_finetune)
            params["train"]["batch_size"] = (params["train"]["batch_size"] // n_classes) * n_classes
            # print("params 3:", params)
                
            # Fine-tuning OB
            train_loader = data_loader(X_train_e, y_train_e, batch_size=params["train"]["batch_size"], num_workers=params["num_workers"], shuffle=True)
            val_loader = data_loader(X_val_e, y_val_e, batch_size=params["train"]["batch_size"], num_workers=params["num_workers"], shuffle=False)
            test_loader = data_loader(X_val_cv, y_val_cv, batch_size=params["train"]["batch_size"], num_workers=params["num_workers"], shuffle=False)
            
            finetune_trainer = BaselineTrainer(params, train_loader, val_loader, test_loader, n_classes, x_dim)
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
            
            cv_scores.append(result[config["train"]["best_metric_name"]])
            
        return np.mean(cv_scores)
    
    # NOTE:
    study = optuna.create_study(direction=config["optuna"]["best_direction"], sampler=optuna.samplers.TPESampler(seed=seed))  # 目標是最大化驗證準確率
    study.optimize(objective, n_trials=config["optuna"]["n_trials"])  # 執行 3 次試驗
    
    # 輸出最佳超參數
    print("Best hyperparameters:", study.best_params)
    
    # NOTE:
    save_path = f'{config["best_hyperparams_dir"]}/{config["dataset"]["dataset_type"]}/{dataset_name}/{config["method_name"]}/seed{config["seed"]}.yaml'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    
    best_params_dict = {
        "search_params": study.best_params,
        "best_value": study.best_value,
        "best_trial": study.best_trial.number,
    }

    with open(save_path, "w") as f:
        yaml.safe_dump(best_params_dict, f, sort_keys=False)
        
    gc.collect()  
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Optimization (PT-RCM + FT-OB)")
    parser.add_argument("--config", type=str, default="configs/optuna/default_optuna_ptrcm_ftob.yaml")
    args = parser.parse_args()

    opt_cfg = load_yaml(args.config)

    torch.set_num_threads(8)
    if opt_cfg["dataset"]["dataset_type"] == "tabular":
        run_ptrcm_ftob_optuna_tabular(opt_cfg)
    elif opt_cfg["dataset"]["dataset_type"] == "image":
        run_ptrcm_ftob_optuna_image(opt_cfg)
    else:
        raise ValueError("Unsupported dataset type")
 
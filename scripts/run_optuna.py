import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import argparse
import yaml
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
import optuna
import gc
from data.data_loader import data_loader, load_tabular_dataset, Sampler
from train.trainer import RankCLTrainer
from src.utils import load_yaml, make_distributions, print_label_distribution, set_seed
from src.factory import get_optuna_params
from data.dataset import load_image_dataset


def run_optuna_tabular(config):
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
            opt_params = {}
            opt_params["search_params"] = get_optuna_params(config["base_method_name"], config.get("use_rankcl", True), trial)
            
            # 合併設定（base 為底，hyper 覆蓋）
            params = config | opt_params
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
                
                val_loader = data_loader(X_val_e, y_val_e, batch_size=params["train"]["batch_size"], num_workers=params["num_workers"], shuffle=False)
                train_loader = data_loader(X_train_e, y_train_e, batch_size=params["train"]["batch_size"], num_workers=params["num_workers"], shuffle=False) 
                train_sampler = Sampler(X_train_e, y_train_e, n_classes, n_samples_per_class=params["train"]["batch_size"]//n_classes)
                test_loader = data_loader(X_val_cv, y_val_cv, batch_size=params["train"]["batch_size"], num_workers=params["num_workers"], shuffle=False)
                
                # net
                x_dim = X_train_e.shape[1]
                             
                trainer = RankCLTrainer(params, train_sampler, train_loader, val_loader, test_loader, n_classes, x_dim)
                # train
                trainer.train()
                # test
                result = trainer.test()
                                
                cv_scores.append(result[config["train"]["best_metric_name"]])
                
            # 計算folds平均分數
            return np.mean(cv_scores)
        
        # 使用 Optuna 進行超參數搜尋
        study = optuna.create_study(direction=config["optuna"]["best_direction"], sampler=optuna.samplers.TPESampler(seed=seed))  # 目標是最大化驗證準確率
        study.optimize(objective, n_trials=config["optuna"]["n_trials"])  # 執行 15 次試驗
        
        # 輸出最佳超參數
        print("Best hyperparameters:", study.best_params)
           
        if distribution is not None:
            save_path = f'{config["best_hyperparams_dir"]}/tabular/{dataset_name}_{distribution}/{config["method_name"]}/seed{config["seed"]}_run{run}.yaml'
        else:
            save_path = f'{config["best_hyperparams_dir"]}/tabular/{dataset_name}/{config["method_name"]}/seed{config["seed"]}_run{run}.yaml'
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
    
def run_optuna_image(config):
    seed = config["seed"]
    set_seed(seed)
    
    dataset_name = config["dataset"]["dataset_name"]

    X_train_full, y_train_full, _, _ = load_image_dataset(dataset_name, seed)
    n_classes = len(np.unique(y_train_full))
    
    skf = StratifiedKFold(n_splits=config["optuna"]["n_folds"], shuffle=True, random_state=seed)
    
    def objective(trial):
        opt_params = {}
        opt_params["search_params"] = get_optuna_params(config["base_method_name"], config.get("use_rankcl", True), trial)
        
        # 合併設定（base 為底，hyper 覆蓋）
        params = config | opt_params
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
            
            val_loader = data_loader(X_val_e, y_val_e, batch_size=params["train"]["batch_size"], num_workers=params["num_workers"], shuffle=False)
            train_loader = data_loader(X_train_e, y_train_e, batch_size=params["train"]["batch_size"], num_workers=params["num_workers"], shuffle=False)
            train_sampler = Sampler(X_train_e, y_train_e, n_classes, n_samples_per_class=params["train"]["batch_size"]//n_classes)
            test_loader = data_loader(X_val_cv, y_val_cv, batch_size=params["train"]["batch_size"], num_workers=params["num_workers"], shuffle=False)
            
            x_dim = X_train_e.shape[1]            
            trainer = RankCLTrainer(params, train_sampler, train_loader, val_loader, test_loader, n_classes, x_dim)
            # train
            trainer.train()
            # test
            result = trainer.test()
                 
            cv_scores.append(result[config["train"]["best_metric_name"]])
            
        return np.mean(cv_scores)
    
    # 使用 Optuna 進行超參數搜尋
    study = optuna.create_study(direction=config["optuna"]["best_direction"], sampler=optuna.samplers.TPESampler(seed=seed))  # 目標是最大化驗證準確率
    study.optimize(objective, n_trials=config["optuna"]["n_trials"])  # 執行 15 次試驗
    
    # 輸出最佳超參數
    print("Best hyperparameters:", study.best_params)
    
    save_path = f'{config["best_hyperparams_dir"]}/image/{dataset_name}/{config["method_name"]}/seed{config["seed"]}.yaml'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    
    best_params_dict = {
        "search_params": study.best_params,
        "best_value": study.best_value,
        "best_trial": study.best_trial.number,
    }

    with open(save_path, "w") as f:
        yaml.safe_dump(best_params_dict, f, sort_keys=False)     
   
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Optimization")
    parser.add_argument("--config", type=str, default="configs/optuna/default_optuna.yaml")
    args = parser.parse_args()

    opt_cfg = load_yaml(args.config)


    if opt_cfg["dataset"]["dataset_type"] == "tabular":
        run_optuna_tabular(opt_cfg)
    elif opt_cfg["dataset"]["dataset_type"] == "image":
        run_optuna_image(opt_cfg)
    else:
        raise ValueError("Unsupported dataset type")
 
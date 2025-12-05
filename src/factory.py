from torch import nn
from dlordinal.output_layers import CLM, StickBreakingLayer
from dlordinal.wrappers import OBDECOCModel
from .model import (BaseTabular, DualHeadTabular, BaseImage, DualHeadImage, OBDECOCHead, PretrainRCMImage, PretrainRCMTabular, TabularNet, ImageNet)
from dlordinal.losses import (
    BetaCrossEntropyLoss,
    TriangularCrossEntropyLoss,
    WKLoss,
    ExponentialRegularisedCrossEntropyLoss,
    OrdinalECOCDistanceLoss
)
from overrides.softlabeling import (
    BinomialCrossEntropyLoss,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from Ordinal_Classifier import Ordinal_Classifier as OC
from mord import LogisticAT  # Logistic model with All-Thresholds (CLM)



def get_optuna_params(base_method_name, use_rankcl, trial):
    if base_method_name=="LinearLayer":
        params = {}
    elif base_method_name=="SoftLabel":
        distribution = trial.suggest_categorical("distribution", ["triangular", "beta", "binomial", "exponential"])
        if distribution == "triangular":
            params = {
                "distribution": distribution,
                "alpha2": trial.suggest_categorical("alpha2", [0.01, 0.05, 0.10]),
                "eta": trial.suggest_categorical("eta", [0.8, 1.0]),
            }
        elif distribution == "beta":
            params = {
                "distribution": distribution,
                "eta": trial.suggest_categorical("eta", [0.8, 1.0]),
            }
        elif distribution == "binomial":
            params = {
                "distribution": distribution,
                "eta": trial.suggest_categorical("eta", [0.8, 1.0]),
            }
        elif distribution == "exponential":
            params = {
                "distribution": distribution,
                "eta": trial.suggest_categorical("eta", [0.8, 1.0]),
                "p": trial.suggest_categorical("p", [1.0, 1.5, 2.0]),
            }
    elif base_method_name=="StickBreaking":
        params = {}
        
    elif base_method_name=="OBDECOC":
        params = {}

    elif base_method_name=="DeepCLM":
        params = {
            "link_function": trial.suggest_categorical("link_function", ['logit','probit', 'cloglog']),
        }
    elif base_method_name=="DeepCLMWK":
        params = {
            "link_function": trial.suggest_categorical("link_function", ['logit','probit', 'cloglog']),
        }
    elif base_method_name == "CLOC":
        params = {}
    else:
        raise ValueError("Invalid method name.")
    
    # 共同參數
    params["lr"] = trial.suggest_categorical("lr", [1e-2, 1e-3, 1e-4])
    
    if use_rankcl:
        params["lambda"] = trial.suggest_categorical("lambda", [0.001, 0.01, 0.1, 1, 10, 100, 1000])
    else:
        print("No lambda tuning.")

    return params

def get_model_loss(num_classes, input_dim, config):
    base_method_name = config["base_method_name"]
    dataset_type = config["dataset"]["dataset_type"]
    if config.get("model") is None:
        model_params = {}
    else:
        model_params = config["model"].get("model_params", {})
    search_params = config.get("search_params", {})
    
    if config.get('use_reweight', False):
        print("Using reweighting")
        class_weights = config['class_weights']
    else:
        print("Not using reweighting")
        class_weights = None
    
    if base_method_name == "LinearLayer":
        prediction_head_name = "Softmax"
        loss_fn_name="cross_entropy"
        
    elif base_method_name == "SoftLabel":
        prediction_head_name = "Softmax"
        loss_fn_name=search_params["distribution"]
    
    elif base_method_name == "StickBreaking":
        prediction_head_name = "StickBreaking"
        loss_fn_name="cross_entropy"
    
    elif base_method_name == "OBDECOC":
        prediction_head_name = "OBDECOC"
        loss_fn_name="ordinal_ecoc"
 
    elif base_method_name == "DeepCLM":
        prediction_head_name = "CLM"
        loss_fn_name="cross_entropy"
        
    elif base_method_name == "DeepCLMWK":
        prediction_head_name = "CLM"
        loss_fn_name="wkloss"
    
    elif base_method_name == "CLOC":
        prediction_head_name = "CE_2layer"
        loss_fn_name="cross_entropy"
        
    else:
        raise ValueError("Invalid method name.")
    
    # NOTE: 預設使用 RankCL 架構
    use_rankcl = config.get("use_rankcl", True)
    
    model = get_model(dataset_type, use_rankcl, num_classes, input_dim, prediction_head_name, model_params, search_params)
    loss_fn = get_loss_fn(num_classes, loss_fn_name, search_params, class_weights)
    
    return model, loss_fn
    

def get_model(dataset_type, use_rankcl, num_classes, input_dim, prediction_head_name, model_params, search_params):
    if use_rankcl:
        if dataset_type == "tabular":
            if prediction_head_name == "OBDECOC":
                print("BaseTabular")
                model = BaseTabular(input_dim=input_dim, **model_params)
            else:
                print("DualHeadTabular")
                model = DualHeadTabular(num_classes, input_dim, **model_params)
        else:
            if prediction_head_name == "OBDECOC":
                print("BaseImage")
                model = BaseImage(**model_params)
            else:
                print("DualHeadImage")
                model = DualHeadImage(num_classes, **model_params)
                
    else:
        if dataset_type == "tabular":
            print("TabularNet")
            model = TabularNet(num_classes, input_dim, **model_params)
        else:
            print("ImageNet")
            model = ImageNet(num_classes)
            
        if prediction_head_name == "OBDECOC":
            print("clf identity")
            model.clf_head = nn.Identity()

    encoded_dim = model.encoded_dim
    
    if prediction_head_name == "Softmax":
        pass
    elif prediction_head_name == "StickBreaking":
        model.clf_head = StickBreakingLayer(encoded_dim, num_classes)
        
    elif prediction_head_name == "CLM":
        clm = CLM(num_classes, link_function=search_params["link_function"])
        has_bias = model.clf_head.bias is not None
        model.clf_head = nn.Sequential(nn.Linear(encoded_dim, 1, bias=has_bias), clm)
    
    elif prediction_head_name == "OBDECOC":
        if use_rankcl:
            model = OBDECOCHead(num_classes=num_classes, base_classifier=model, base_n_outputs=encoded_dim)
        else:
            model = OBDECOCModel(num_classes=num_classes, base_classifier=model, base_n_outputs=encoded_dim)
    elif prediction_head_name == "CE_2layer":
        model.clf_head = nn.Sequential(
            nn.Linear(encoded_dim, encoded_dim//2),
            nn.ReLU(),
            nn.Linear(encoded_dim//2, num_classes)
        )
    
    else:
        raise ValueError("Invalid output layer name.")
    
    return model


def get_loss_fn(num_classes, loss_fn_name, search_params, class_weights):
    if class_weights is not None:
        print(f"Class weights: {class_weights}")
        
    if loss_fn_name == "cross_entropy":
        return nn.CrossEntropyLoss(weight=class_weights)
    
    elif loss_fn_name == "triangular":
        return TriangularCrossEntropyLoss(num_classes=num_classes, alpha2=search_params["alpha2"], eta=search_params["eta"], weight=class_weights)
    
    elif loss_fn_name == "beta":
        return BetaCrossEntropyLoss(num_classes=num_classes, eta=search_params["eta"], weight=class_weights)

    elif loss_fn_name == "binomial":
        return BinomialCrossEntropyLoss(num_classes=num_classes, eta=search_params["eta"], weight=class_weights)

    elif loss_fn_name == "ordinal_ecoc":
        return OrdinalECOCDistanceLoss(num_classes=num_classes, weights=class_weights)
    
    elif loss_fn_name == "exponential":
        return ExponentialRegularisedCrossEntropyLoss(num_classes=num_classes, eta=search_params["eta"], p=search_params["p"], weight=class_weights)
    
    elif loss_fn_name == "wkloss":
        return WKLoss(num_classes=num_classes, weight=class_weights)
    
    else:
        raise ValueError("Invalid loss function name.")
    

def get_optuna_params_ml(base_method_name, trial):
    if base_method_name == "Decision Tree":
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        }

    elif base_method_name == "Random Forest":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300),
            "max_depth": trial.suggest_int("max_depth", 5, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        }

    elif base_method_name == "XGBoost":
        params = {
            "tree_method": "hist",
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        }
    elif base_method_name == "LightGBM":
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        }

    elif base_method_name == "MLP":
        params = {
            "hidden_layer_sizes": (128, 64),
            "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-2, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
        }
    elif base_method_name == "Logistic Regression":
        params = {
            "C": trial.suggest_float("C", 1e-3, 1e2, log=True),  
            "solver": "lbfgs",  
            "multi_class": "multinomial",
            "max_iter": 400,
            "penalty": "l2",  
        }
        
    elif base_method_name == "Ordinal Tree":
        clf_name = trial.suggest_categorical("clf", ["decision_tree", "random_forest"])
        
        if clf_name == "decision_tree":
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
            }
        elif clf_name == "random_forest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 300),
                "max_depth": trial.suggest_int("max_depth", 5, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            }
            
        params["clf"] = clf_name    
        
    elif base_method_name == "CLM":
        params = {
            "alpha": trial.suggest_float("alpha", 1e-3, 100, log=True)
        }
    
    return params 

def get_model_ml(base_method_name, params):
    if base_method_name == "Decision Tree":
        model = DecisionTreeClassifier(**params)

    elif base_method_name == "Random Forest":
        model = RandomForestClassifier(**params)

    elif base_method_name == "XGBoost":
        model = XGBClassifier(**params, eval_metric="logloss")

    elif base_method_name == "LightGBM":
        model = LGBMClassifier(**params)
        
    elif base_method_name == "MLP":
        model = MLPClassifier(max_iter=400, **params)
        
    elif base_method_name == "Logistic Regression":
        model = LogisticRegression(**params)
        
    elif base_method_name == "Ordinal Tree":
        if params["clf"] == "decision_tree":
            model_params = {k: v for k, v in params.items() if k != "clf"}
            clf = DecisionTreeClassifier(**model_params)
        elif params["clf"] == "random_forest":
            model_params = {k: v for k, v in params.items() if k != "clf"}
            clf = RandomForestClassifier(**model_params)
        model = OC.OrdinalClassifier(clf)
        
    elif base_method_name == "CLM":
        model = LogisticAT(**params)
    
    return model
    
def get_pretrain_model(input_dim, config):
    dataset_type = config["dataset"]["dataset_type"]
    if config.get("model") is None:
        model_params = {}
    else:
        model_params = config["model"].get("model_params", {})
        
    if dataset_type == "tabular":
        model = PretrainRCMTabular(input_dim, **model_params)
    else:
        model = PretrainRCMImage(**model_params) 
    
    return model


def get_pretrain_optuna_params(trial):
    params = {}
    params["pretrain_lr"] = trial.suggest_categorical("pretrain_lr", [1e-2, 1e-3, 1e-4])
    return params
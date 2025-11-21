from torch import nn
from dlordinal.output_layers import CLM
from dlordinal.output_layers import StickBreakingLayer
from .model import (BaseTabular, DualHeadTabular, BaseImage, DualHeadImage, OBDECOCHead, TabularNet, ImageNet)
from dlordinal.losses import (
    BetaCrossEntropyLoss,
    TriangularCrossEntropyLoss,
    WKLoss,
    ExponentialRegularisedCrossEntropyLoss,
    OrdinalECOCDistanceLoss
)
from dlordinal_change.softlabeling import (
    BinomialCrossEntropyLoss,
)


def get_optuna_params(deep_ordinal_method, trial):
    if deep_ordinal_method=="LinearLayer":
        params = {}
    elif deep_ordinal_method=="SoftLabel":
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
    elif deep_ordinal_method=="StickBreaking":
        params = {}
        
    elif deep_ordinal_method=="OBDECOC":
        params = {}

    elif deep_ordinal_method=="DeepCLM":
        params = {
            "link_function": trial.suggest_categorical("link_function", ['logit','probit', 'cloglog']),
        }
    elif deep_ordinal_method=="DeepCLMWK":
        params = {
            "link_function": trial.suggest_categorical("link_function", ['logit','probit', 'cloglog']),
        }
    elif deep_ordinal_method == "CLOC":
        params = {}
    else:
        raise ValueError("Invalid method name.")
    
    # 共同參數
    params["lr"] = trial.suggest_categorical("lr", [1e-2, 1e-3, 1e-4])
    params["lambda"] = trial.suggest_categorical("lambda", [0.001, 0.01, 0.1, 1, 10, 100, 1000])

    return params

def get_model_loss(num_classes, input_dim, config):
    deep_ordinal_method = config["deep_ordinal_method"]
    dataset_type = config["dataset"]["dataset_type"]
    model_params = config["model"].get("model_params", {})
    search_params = config.get("search_params", {})
    
    if config.get('use_reweight', False):
        print("Using reweighting")
        class_weights = config['class_weights']
    else:
        print("Not using reweighting")
        class_weights = None
    
    if deep_ordinal_method == "LinearLayer":
        prediction_head_name = "Softmax"
        loss_fn_name="cross_entropy"
        
    elif deep_ordinal_method == "SoftLabel":
        prediction_head_name = "Softmax"
        loss_fn_name=search_params["distribution"]
    
    elif deep_ordinal_method == "StickBreaking":
        prediction_head_name = "StickBreaking"
        loss_fn_name="cross_entropy"
    
    elif deep_ordinal_method == "OBDECOC":
        prediction_head_name = "OBDECOC"
        loss_fn_name="ordinal_ecoc"
 
    elif deep_ordinal_method == "DeepCLM":
        prediction_head_name = "CLM"
        loss_fn_name="cross_entropy"
        
    elif deep_ordinal_method == "DeepCLMWK":
        prediction_head_name = "CLM"
        loss_fn_name="wkloss"
    
    elif deep_ordinal_method == "CLOC":
        prediction_head_name = "CE_2layer"
        loss_fn_name="cross_entropy"
        
    else:
        raise ValueError("Invalid method name.")
    
    # NOTE: 預設使用 RankCL 架構
    use_rankcl = config.get("use_rankcl", True)
    method_type = config.get("method_type", None)
    
    model = get_model(dataset_type, use_rankcl, method_type, num_classes, input_dim, prediction_head_name, model_params, search_params)
    loss_fn = get_loss_fn(num_classes, loss_fn_name, search_params, class_weights)
    
    return model, loss_fn
    

def get_model(dataset_type, use_rankcl, method_type, num_classes, input_dim, prediction_head_name, model_params, search_params):
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
                
    elif method_type == 'deep':
        if dataset_type == "tabular":
            print("TabularNet")
            model = TabularNet(num_classes, input_dim, **model_params)
        else:
            print("ImageNet")
            model = ImageNet(num_classes)
            
        if prediction_head_name == "OBDECOC":
            print("clf identity")
            model.clf_head = nn.Identity()
    else:
        raise ValueError("Invalid method type.")

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
        model = OBDECOCHead(num_classes=num_classes, base_classifier=model, base_n_outputs=encoded_dim)
    
    elif prediction_head_name == "CE_2layer":
        model.clf_head = nn.Sequential(
            nn.Linear(encoded_dim, encoded_dim//2),
            nn.ReLU(),
            nn.Linear(encoded_dim//2, num_classes)
        )
    
    else:
        raise ValueError("Invalid output layer name.")
    
    return model

# def get_model(dataset_type, use_rankcl, method_type, num_classes, input_dim, prediction_head_name, model_params, search_params):
#     if use_rankcl:
#         return get_model_rankcl(dataset_type, num_classes, input_dim, prediction_head_name, model_params, search_params)
#     elif method_type == 'deep':
#         return get_model_deep(dataset_type, num_classes, input_dim, prediction_head_name, model_params, search_params)
#     else:
#         raise ValueError("Invalid method type.")
     
# def get_model_rankcl(dataset_type, num_classes, input_dim, prediction_head_name, model_params, search_params):
    
#     if dataset_type == "tabular":
#         if prediction_head_name == "OBDECOC":
#             base = BaseTabular(input_dim=input_dim, **model_params)
#             encoded_dim = base.encoded_dim
#         else:
#             model = DualHeadTabular(num_classes, input_dim, **model_params)
#             encoded_dim = model.encoded_dim
#     else:
#         if prediction_head_name == "OBDECOC":
#             base = BaseImage(**model_params)
#             encoded_dim = base.encoded_dim
#         else:
#             model = DualHeadImage(num_classes, **model_params)
#             encoded_dim = model.encoded_dim

#     if prediction_head_name == "Softmax":
#         pass
#     elif prediction_head_name == "StickBreaking":
#         model.clf_head = StickBreakingLayer(encoded_dim, num_classes)
        
#     elif prediction_head_name == "CLM":
#         clm = CLM(num_classes, link_function=search_params["link_function"])
#         has_bias = model.clf_head.bias is not None
#         model.clf_head = nn.Sequential(nn.Linear(encoded_dim, 1, bias=has_bias), clm)
    
#     elif prediction_head_name == "OBDECOC":
#         model = OBDECOCHead(num_classes=num_classes, base_classifier=base, base_n_outputs=encoded_dim)
    
#     elif prediction_head_name == "CE_2layer":
#         model.clf_head = nn.Sequential(
#             nn.Linear(encoded_dim, encoded_dim//2),
#             nn.ReLU(),
#             nn.Linear(encoded_dim//2, num_classes)
#         )
    
#     else:
#         raise ValueError("Invalid output layer name.")
    
#     return model


def get_loss_fn(num_classes, loss_fn_name, search_params, class_weights):
    
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
    

def get_optuna_params_ml(method_name, trial):
    if method_name == "decision_tree":
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        }

    elif method_name == "random_forest":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300),
            "max_depth": trial.suggest_int("max_depth", 5, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        }

    elif method_name == "xgboost":
        params = {
            "tree_method": "hist",
            "device": "cuda",
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        }

    elif method_name == "gradient_boosting":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        }

    elif method_name == "lightgbm":
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        }

    elif method_name == "catboost":
        params = {
            "task_type": "GPU",
            "iterations": trial.suggest_int("iterations", 100, 300),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "depth": trial.suggest_int("depth", 3, 10),
            "verbose": 0,
        }

    elif method_name == "MLP":
        params = {
            "hidden_layer_sizes": (128, 64),
            "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-2, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
        }
    elif method_name == "logistic_regression":
        params = {
            "C": trial.suggest_float("C", 1e-3, 1e2, log=True),  
            "solver": "lbfgs",  
            "multi_class": "multinomial",
            "max_iter": 400,
            "penalty": "l2",  
        }
        
    elif method_name == "ordinal_classfier":
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
        
    elif method_name == "CLM":
        params = {
            "alpha": trial.suggest_float("alpha", 1e-3, 100, log=True)
        }
    
    elif method_name == "OrdinalRidge":
        params = {
            "alpha": trial.suggest_float("alpha", 1e-3, 100, log=True)
        }
    
    return params 

def get_model_ml(method_name, params):
    if method_name == "decision_tree":
        model = DecisionTreeClassifier(**params)

    elif method_name == "random_forest":
        model = RandomForestClassifier(**params)

    elif method_name == "xgboost":
        model = XGBClassifier(**params, eval_metric="logloss")

    elif method_name == "gradient_boosting":
        model = GradientBoostingClassifier(**params)

    elif method_name == "lightgbm":
        model = LGBMClassifier(**params)

    elif method_name == "catboost":
        model = CatBoostClassifier(**params)

    elif method_name == "MLP":
        model = MLPClassifier(max_iter=400, **params)
        
    elif method_name == "logistic_regression":
        model = LogisticRegression(**params)
        
    elif method_name == "ordinal_classfier":
        if params["clf"] == "decision_tree":
            model_params = {k: v for k, v in params.items() if k != "clf"}
            clf = DecisionTreeClassifier(**model_params)
        elif params["clf"] == "random_forest":
            model_params = {k: v for k, v in params.items() if k != "clf"}
            clf = RandomForestClassifier(**model_params)
        model = OC.OrdinalClassifier(clf)
        
    elif method_name == "CLM":
        model = LogisticAT(**params)
    
    elif method_name == "OrdinalRidge":
        model = OrdinalRidge(**params)
    return model
    

from torch import nn
# from dlordinal.dropout import HybridDropout, HybridDropoutContainer
from dlordinal.output_layers import CLM
from dlordinal.output_layers import StickBreakingLayer
from .model import (BaseTabular, DualHeadTabular, BaseImage, DualHeadImage, OBDECOCHead)
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
    # elif deep_ordinal_method==("HybridDropout"):
    #     params = {}
    elif deep_ordinal_method=="CLM":
        params = {
            "link_function": trial.suggest_categorical("link_function", ['logit','probit', 'cloglog']),
        }
    elif deep_ordinal_method=="CLMWK":
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
        loss_fn_name="ordinal_ecoc_noWeights"
    
    # elif deep_ordinal_method == "HybridDropout":
    #     prediction_head_name = "hybrid_dropout"
    #     loss_fn_name="cross_entropy"
    
    elif deep_ordinal_method == "CLM":
        prediction_head_name = "CLM"
        loss_fn_name="cross_entropy"
        
    elif deep_ordinal_method == "CLMWK":
        prediction_head_name = "CLM"
        loss_fn_name="wkloss"
    
    elif deep_ordinal_method == "CLOC":
        prediction_head_name = "CE_2layer"
        loss_fn_name="cross_entropy"
        
    else:
        raise ValueError("Invalid method name.")
    
    model = get_model(dataset_type, num_classes, input_dim, prediction_head_name, model_params, search_params)
    loss_fn = get_loss_fn(num_classes, loss_fn_name, search_params)
    
    return model, loss_fn

def get_model(dataset_type, num_classes, input_dim, prediction_head_name, model_params, search_params):
    
    if dataset_type == "tabular":
        if prediction_head_name == "OBDECOC":
            base = BaseTabular(input_dim=input_dim, **model_params)
            encoded_dim = base.encoded_dim
        else:
            model = DualHeadTabular(num_classes, input_dim, **model_params)
            encoded_dim = model.encoded_dim
    else:
        if prediction_head_name == "OBDECOC":
            base = BaseImage(**model_params)
            encoded_dim = base.encoded_dim
        else:
            model = DualHeadImage(num_classes, **model_params)
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
        model = OBDECOCHead(num_classes=num_classes, base_classifier=base, base_n_outputs=encoded_dim)
    
    elif prediction_head_name == "CE_2layer":
        model.clf_head = nn.Sequential(
            nn.Linear(encoded_dim, encoded_dim//2),
            nn.ReLU(),
            nn.Linear(encoded_dim//2, num_classes)
        )
            
    # elif prediction_head_name == "hybrid_dropout":
    #     model.clf_head = nn.Sequential(
    #         nn.Linear(encoded_dim, encoded_dim//2),
    #         HybridDropout(),
    #         nn.Linear(encoded_dim//2, num_classes),
    #     )
    #     model = HybridDropoutContainer(model)
    
    else:
        raise ValueError("Invalid output layer name.")
    
    return model


def get_loss_fn(num_classes, loss_fn_name, search_params):
    
    if loss_fn_name == "cross_entropy":
        return nn.CrossEntropyLoss()
    
    elif loss_fn_name == "triangular":
        return TriangularCrossEntropyLoss(num_classes=num_classes, alpha2=search_params["alpha2"], eta=search_params["eta"])
    
    elif loss_fn_name == "beta":
        return BetaCrossEntropyLoss(num_classes=num_classes, eta=search_params["eta"])

    elif loss_fn_name == "binomial":
        return BinomialCrossEntropyLoss(num_classes=num_classes, eta=search_params["eta"])

    elif loss_fn_name == "ordinal_ecoc_noWeights":
        return OrdinalECOCDistanceLoss(num_classes=num_classes)
    
    elif loss_fn_name == "exponential":
        return ExponentialRegularisedCrossEntropyLoss(num_classes=num_classes, eta=search_params["eta"], p=search_params["p"])
    
    elif loss_fn_name == "wkloss":
        return WKLoss(num_classes=num_classes)
    
    else:
        raise ValueError("Invalid loss function name.")
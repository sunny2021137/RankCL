# dl ordinal metrics
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    mean_absolute_error,
    f1_score
)
from dlordinal.metrics import amae, mmae
import numpy as np

def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int):

    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)

    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)

    labels = range(0, num_classes)

    # Metrics calculation
    amae_ = amae(y_true, y_pred)
    mmae_ = mmae(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', labels=labels)
    ms = minimum_sensitivity(y_true, y_pred, labels=labels)
    mae = mean_absolute_error(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=labels)
    
    metrics = {
        "AMAE": amae_,
        "MMAE": mmae_,
        "f1-score(macro avg)": f1_macro,
        "MS": ms,
        "MAE": mae,
        "CCR": acc,
        "QWK": qwk,
    }

    return metrics
    

def _compute_sensitivities(y_true, y_pred, labels=None):
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)

    conf_mat = confusion_matrix(y_true, y_pred, labels=labels)

    sum = np.sum(conf_mat, axis=1)
    mask = np.eye(conf_mat.shape[0], conf_mat.shape[1])
    correct = np.sum(conf_mat * mask, axis=1)
    sensitivities = correct / sum

    sensitivities = sensitivities[~np.isnan(sensitivities)]

    return sensitivities


def minimum_sensitivity(y_true, y_pred, labels=None):
    return np.min(_compute_sensitivities(y_true, y_pred, labels=labels))
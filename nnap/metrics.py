from typing import List

import numpy as np
from sklearn import metrics
from tensorflow.keras.callbacks import History


def calculate_sensitivity(tp, fn):
    return tp / (tp + fn)


def calculate_specificity(tn, fp):
    return tn / (tn + fp)


def calculate_accuracy(tp, tn, y):
    return (tp + tn) / len(y)


def calculate_q_value(sensitivity, specificity):
    return (sensitivity + specificity) / 2


def calculate_f1_score(tp, fn, fp):
    return 2 * tp / (2 * tp + fn + fp)


def calculate_metrics(y, y_):
    cm = metrics.confusion_matrix(y, np.round(y_))
    tn, fp, fn, tp = cm.ravel()
    sensitivity = calculate_sensitivity(tp, fn)
    specificity = calculate_specificity(tn, fp)
    accuracy = calculate_accuracy(tp, tn, y)
    q_value = calculate_q_value(sensitivity, specificity)
    f1_score = calculate_f1_score(tp, fn, fp)
    mcc = metrics.matthews_corrcoef(y, np.round(y_))
    auc = metrics.roc_auc_score(y, y_)
    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "accuracy": accuracy,
        "q_value": q_value,
        "f1_score": f1_score,
        "mcc": mcc,
        "auc": auc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def process_training_logs(training_logs: List[History]):
    metrics = training_logs[0].history.keys()
    metric_stats = {}
    for metric in metrics:
        min_vals = []
        max_vals = []
        for log in training_logs:
            metric_values = log.history[metric]
            min_vals.append(min(metric_values))
            max_vals.append(max(metric_values))
        metric_stats[metric] = {
            "max_mean": np.mean(max_vals),
            "min_mean": np.mean(min_vals),
        }
    return metric_stats







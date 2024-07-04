"""Functions for evaluating model performance."""
from enum import Enum
from typing import Any

import numpy as np
from numpy import floating, ndarray


class EvaluationMetrics(Enum):
    LOG_LOSS = "log_loss"
    BRIER_SCORE = "brier_score"
    BIAS = "bias"
    RMSE = "rmse"


def log_loss(y_true: ndarray, y_pred: ndarray) -> floating[Any]:
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def brier_score(y_true: ndarray, y_pred: ndarray) -> floating[Any]:
    return np.mean((y_true - y_pred) ** 2)


def bias(y_true: ndarray, y_pred: ndarray) -> floating[Any]:
    return np.mean(y_pred - y_true)


def rmse(y_true: ndarray, y_pred: ndarray) -> floating[Any]:
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


metric_to_function = {
    EvaluationMetrics.LOG_LOSS: log_loss,
    EvaluationMetrics.BRIER_SCORE: brier_score,
    EvaluationMetrics.BIAS: bias,
    EvaluationMetrics.RMSE: rmse,
}

import logging
from typing import Callable

import torch
from sklearn.metrics import f1_score, recall_score, precision_score


def calculate_basic_metrics(
    labels_cpu: torch.Tensor,
    predictions_flat: torch.Tensor,
    evaluation_metric_weighting: str,
    logger: Callable[[str], None] = logging.info,
) -> float:
    f_score = f1_score(
        labels_cpu,
        predictions_flat,
        average=evaluation_metric_weighting,
    )
    recall = recall_score(
        labels_cpu,
        predictions_flat,
        average=evaluation_metric_weighting,
    )
    precision = precision_score(
        labels_cpu,
        predictions_flat,
        average=evaluation_metric_weighting,
    )
    logger(f"Evaluation metrics: {f_score=}, {recall=}, {precision=}")
    return f_score

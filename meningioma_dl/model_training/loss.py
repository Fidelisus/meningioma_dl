from collections import Counter
from typing import List, Callable

import numpy as np
import torch
from torch import nn

from meningioma_dl.models.resnet import ResNet


def get_loss_function_class_weights(labels: List[int]) -> np.array:
    counts = Counter(labels)
    return np.array(list(counts.values())) / sum(list(counts.values()))


def calculate_loss_using_proximal_term(
    global_params: ResNet,
    model: ResNet,
    labels_torch_format: torch.Tensor,
    predictions: torch.Tensor,
    loss_function: Callable,
    proximal_mu: float,
) -> torch.Tensor:
    proximal_term = 0.0
    for local_weights, global_weights in zip(
        model.parameters(), global_params.parameters()
    ):
        proximal_term += (local_weights - global_weights).norm(2)
    return (
        loss_function(predictions, labels_torch_format)
        + (proximal_mu / 2.0) * proximal_term
    )


def get_loss_function(
    labels_train: list[int],
    labels_validation: list[int],
    evaluation_metric_weighting: str,
):
    if evaluation_metric_weighting == "weighted":
        loss_function_weighting = torch.tensor(
            get_loss_function_class_weights(labels_train + labels_validation)
        ).to(torch.float64)
    elif evaluation_metric_weighting == "macro":
        loss_function_weighting = None
    else:
        raise ValueError(f"{evaluation_metric_weighting=} not supported.")
    return nn.CrossEntropyLoss(
        weight=loss_function_weighting,
    )

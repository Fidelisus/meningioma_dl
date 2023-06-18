from collections import Counter

import numpy as np
import torch
import torch_directml
from torch import nn


def select_device(device="") -> torch.device:
    if device.lower() == "cuda":
        if not torch.cuda.is_available():
            print("torch.cuda not available")
            return torch.device("cpu")
        else:
            return torch.device("cuda")
    if device.lower() == "dml":
        return torch_directml.device(torch_directml.default_device())
    else:
        return torch.device("cpu")


def get_loss_function_class_weights(labels: list[int]) -> np.array:
    counts = Counter(labels)
    return np.array(list(counts.values())) / sum(list(counts.values()))


def one_hot_encode_labels(labels: np.array) -> torch.Tensor:
    labels_onehot = nn.functional.one_hot(labels - 1, num_classes=3).float()
    return labels_onehot

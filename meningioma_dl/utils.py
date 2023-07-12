import logging
from collections import Counter

import numpy as np
import torch
import torch_directml
from torch import nn


def select_device(device="") -> torch.device:
    if device.lower() == "cuda":
        if not torch.cuda.is_available():
            logging.warning("torch.cuda not available")
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


def setup_logging() -> None:
    root_logger = logging.getLogger()
    log_formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s | %(filename)s %(funcName)s:%(lineno)d | %(message)s",
        "%m-%d %H:%M:%S",
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.INFO)

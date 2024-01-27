import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import numpy as np
import shortuuid
import torch

from meningioma_dl.config import Config


def select_device(device="") -> torch.device:
    if device.lower() == "cuda":
        if not torch.cuda.is_available():
            logging.warning("torch.cuda not available")
            return torch.device("cpu")
        else:
            return torch.device("cuda")
    # if device.lower() == "dml":
    #     return torch_directml.device(torch_directml.default_device())
    # else:
    return torch.device("cpu")


def get_loss_function_class_weights(labels: List[int]) -> np.array:
    counts = Counter(labels)
    return np.array(list(counts.values())) / sum(list(counts.values()))


def one_hot_encode_labels(labels: np.array) -> torch.Tensor:
    """
    For now not used, asa not needed by the loss function
    """
    # This function returns core dumped on the cluster, so we cannot use it
    # labels_onehot = nn.functional.one_hot(labels - 1, num_classes=num_classes).float()

    labels = labels.cpu().numpy().astype(int)
    one_hot_encoded = np.zeros((labels.size, labels.max() + 1))
    one_hot_encoded[np.arange(labels.size), labels] = 1
    labels_onehot = torch.Tensor(one_hot_encoded, device="cpu").to(torch.int64)
    return labels_onehot


def setup_logging(log_file_path: Optional[Path]) -> None:
    root_logger = logging.getLogger()
    log_formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s | %(filename)s %(funcName)s:%(lineno)d | %(message)s",
        "%m-%d %H:%M:%S",
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    if log_file_path is not None:
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)

    root_logger.setLevel(logging.INFO)


def generate_run_id() -> str:
    return f"{datetime.now().strftime('%d-%m-%y_%H-%M-%S')}_{shortuuid.uuid()}"


def setup_run(
    env_file_path: Optional[str],
    run_id: Optional[str],
    manual_seed: int,
    device_name: str,
):
    if run_id is None:
        run_id = generate_run_id()
    if env_file_path is not None:
        Config.load_env_variables(env_file_path, run_id)
        setup_logging(Config.log_file_path)
    device = get_device(device_name)
    torch.manual_seed(manual_seed)
    return device


def get_device(device_name) -> torch.device:
    device = select_device(device_name)
    logging.info("Devices available:")
    torch.cuda.is_available()
    for device_number in range(torch.cuda.device_count()):
        logging.info(torch.cuda.get_device_name(device_number))
    return device

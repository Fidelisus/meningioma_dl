import logging
from datetime import datetime
from typing import Literal

import shortuuid
import torch

from meningioma_dl.config import Config


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


def generate_run_id() -> str:
    return f"{datetime.now().strftime('%d-%m-%y_%H-%M-%S')}_{shortuuid.uuid()}"


def setup_run(
    env_file_path: str,
    manual_seed: int,
    device_name: Literal["cpu", "cuda"],
    cv_fold: int,
) -> Config:
    setup_logging()
    torch.manual_seed(manual_seed)
    return Config.from_env_variables(
        env_file_path=env_file_path, cv_fold=cv_fold, device=get_device(device_name)
    )


def get_device(device_name: Literal["cpu", "cuda"]) -> torch.device:
    if device_name.lower() == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("torch.cuda not available")
        return torch.device("cuda")
    return torch.device("cpu")


def setup_flower_logger():
    import flwr as fl
    from flwr.common.logger import LOGGER_NAME

    fl.common.logger.configure(identifier="FL")
    flower_logger = logging.getLogger(LOGGER_NAME)
    flower_logger.propagate = False

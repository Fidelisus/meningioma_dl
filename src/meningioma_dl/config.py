import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import torch
from dotenv import load_dotenv
from typing_extensions import Self


class TaskType(Enum):
    TRAIN: str = "train"
    TEST: str = "test"


@dataclass
class Config:
    # directories
    train_labels_file_path: Path
    validation_labels_file_path: Path
    test_labels_file_path: Path
    data_directory: Path
    pretrained_models_directory: Path
    saved_models_directory: Path

    # visualizations
    visualizations_directory: Path

    # model
    device: torch.device
    test_size: float = 0.2
    validation_size: float = 0.2
    save_intermediate_models: bool = False

    # logging
    log_file_path: Optional[Path] = None

    @classmethod
    def from_env_variables(
        cls, env_file_path: str, cv_fold: int, device: torch.device
    ) -> Self:
        load_dotenv(env_file_path, verbose=True)

        data_directory = Path(os.environ["DATA_DIR"])
        pretrained_models_directory = Path(os.environ["PRETRAINED_MODELS_DIR"])

        visualizations_directory = Path(os.environ["VISUALIZATIONS_DIRECTORY"])
        visualizations_directory.mkdir(parents=True, exist_ok=True)

        saved_models_directory = Path(os.environ["SAVED_MODELS_DIR"])
        saved_models_directory.mkdir(parents=True, exist_ok=True)

        labels_directory = Path(os.environ["LABELS_DIR"])
        train_labels_file_path = labels_directory.joinpath(
            f"train_labels_{cv_fold}.tsv"
        )
        validation_labels_file_path = labels_directory.joinpath(
            f"validation_labels_{cv_fold}.tsv"
        )
        test_labels_file_path = labels_directory.joinpath("test_labels.tsv")

        return Config(
            train_labels_file_path=train_labels_file_path,
            validation_labels_file_path=validation_labels_file_path,
            test_labels_file_path=test_labels_file_path,
            data_directory=data_directory,
            pretrained_models_directory=pretrained_models_directory,
            saved_models_directory=saved_models_directory,
            visualizations_directory=visualizations_directory,
            device=device,
        )

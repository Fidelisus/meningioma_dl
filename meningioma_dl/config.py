import logging
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


class TaskType(Enum):
    TRAIN: str = "train"
    TEST: str = "test"


@dataclass
class Config:
    # directories
    train_labels_file_path: Path
    validation_labels_file_path: Path
    test_labels_file_path: Path
    labels_directory: Path
    data_directory: Path
    pretrained_models_directory: Path
    saved_models_directory: Path

    # visualizations
    visualizations_directory: Path

    # model
    test_size: float = 0.2
    validation_size: float = 0.2
    save_intermediate_models: bool = False

    # logging
    log_file_path: Optional[Path] = None

    loaded: bool = False

    @classmethod
    def load_env_variables(
        cls, env_file_path: str, run_id: str, validation_fold: Optional[int] = None
    ) -> None:
        if cls.loaded:
            raise RuntimeError(f"Config already loaded")
        cls.loaded = True

        load_dotenv(env_file_path, verbose=True)

        cls.data_directory = Path(os.environ["DATA_DIR"])
        cls.labels_directory = Path(os.environ["LABELS_DIR"])
        cls.results_storage_directory = Path(os.environ["RESULTS_STORAGE_DIR"])
        cls.pretrained_models_directory = Path(os.environ["PRETRAINED_MODELS_DIR"])
        cls.saved_models_directory = Path(os.environ["SAVED_MODELS_DIR"])

        if validation_fold is None:
            cls.train_labels_file_path: Path = cls.labels_directory.joinpath(
                "train_labels.tsv"
            )
            cls.validation_labels_file_path: Path = cls.labels_directory.joinpath(
                "validation_labels.tsv"
            )
        else:
            cls.train_labels_file_path: Path = cls.labels_directory.joinpath(
                f"train_labels_{validation_fold}.tsv"
            )
            cls.validation_labels_file_path: Path = cls.labels_directory.joinpath(
                f"validation_labels_{validation_fold}.tsv"
            )
        cls.test_labels_file_path: Path = cls.labels_directory.joinpath(
            "test_labels.tsv"
        )

        cls.pretrained_models_directory.mkdir(parents=True, exist_ok=True)
        cls.saved_models_directory.mkdir(parents=True, exist_ok=True)

        logs_directory = Path(os.environ.get("LOGS_DIR", None))
        if logs_directory:
            cls.log_file_path = logs_directory.joinpath(run_id, "logs.log")
            cls.log_file_path.parent.mkdir(parents=True, exist_ok=True)

        cls.visualizations_directory: Path = logs_directory
        cls.visualizations_directory.mkdir(parents=True, exist_ok=True)

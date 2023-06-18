from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class TaskType(Enum):
    TRAIN: str = "train"
    TEST: str = "test"


@dataclass
class Config:
    # TODO different classes
    # model
    test_size: float = 0.2
    validation_size: float = 0.1
    random_seed: int = 123

    # directories
    data_root_directory: Path = Path("Z:/data/meningioma/nifti")
    ci_run_data_root_directory: Path = Path("../data/scans")
    labels_dir: Path = Path(__file__).parents[1].joinpath("data", "labels")
    labels_file_path: Path = labels_dir.joinpath("labels.tsv")
    train_labels_file_path: Path = labels_dir.joinpath("train_labels.tsv")
    validation_labels_file_path: Path = labels_dir.joinpath("validation_labels.tsv")
    test_labels_file_path: Path = labels_dir.joinpath("test_labels.tsv")
    ci_run_labels_file_path: Path = labels_dir.joinpath("ci_run_labels.tsv")

    # optuna
    results_storage_directory: Path = Path(__file__).parents[1].joinpath("optuna")
    results_storage_directory.mkdir(parents=True, exist_ok=True)
    optuna_database_directory: str = (
        f"sqlite:///{results_storage_directory.joinpath('optuna_store.db')}"
    )

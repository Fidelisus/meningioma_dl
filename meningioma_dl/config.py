from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class TaskType(Enum):
    TRAIN: str = "train"
    TEST: str = "test"


@dataclass
class Config:
    # model
    test_size: float = 0.2
    validation_size: float = 0.1
    random_seed: int = 123

    # directories
    images_directory: Path = Path("Z:/data/meningioma/nifti")
    # TODO TODO change everything to absolute paths using env var
    local_data_root: Path = Path(
        "C:/Users/Lenovo/Desktop/meningioma_project/meningioma_dl/data/"
    )
    ci_images_directory: Path = local_data_root.joinpath("scans")
    labels_dir: Path = local_data_root.joinpath("labels")
    labels_file_path: Path = labels_dir.joinpath("labels.tsv")
    train_labels_file_path: Path = labels_dir.joinpath("train_labels.tsv")
    validation_labels_file_path: Path = labels_dir.joinpath("validation_labels.tsv")
    test_labels_file_path: Path = labels_dir.joinpath("test_labels.tsv")
    ci_run_labels_file_path: Path = labels_dir.joinpath("ci_run_labels.tsv")

    # visualizations
    visualizations_directory: Path = local_data_root.joinpath("viz")
    visualizations_directory.mkdir(parents=True, exist_ok=True)

    # optuna
    results_storage_directory: Path = local_data_root.joinpath("optuna")
    results_storage_directory.mkdir(parents=True, exist_ok=True)
    optuna_database_directory: str = (
        f"sqlite:///{results_storage_directory.joinpath('optuna_store.db')}"
    )

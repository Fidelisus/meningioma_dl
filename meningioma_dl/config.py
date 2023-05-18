from dataclasses import dataclass
from pathlib import Path


@dataclass
class Configuration:
    test_size: float = 0.2
    random_seed: int = 123
    base_dir: Path = Path(__file__).parents[1].joinpath("data")
    labels_dir: Path = base_dir.joinpath("labels")
    labels_file_path: Path = labels_dir.joinpath("labels.tsv")
    train_labels_file_path: Path = labels_dir.joinpath("train_labels.tsv")
    test_labels_file_path: Path = labels_dir.joinpath("test_labels.tsv")

from pathlib import Path
from typing import List, Tuple, Optional, Set, Dict

import pandas as pd

from meningioma_dl.config import Config


def get_training_samples_df() -> pd.DataFrame:
    return get_samples_df(Config.train_labels_file_path)


def get_test_samples_df() -> pd.DataFrame:
    return get_samples_df(Config.test_labels_file_path)


def get_samples_df(labels_file: Path) -> pd.DataFrame:
    labels = pd.read_csv(labels_file, sep="\t")
    return labels


def get_images_with_labels(
    data_root_directory,
    labels_file_path,
    class_mapping: Optional[Dict[int, int]] = None,
) -> Tuple[List[str], List[str], List[int]]:
    samples_df = get_samples_df(labels_file_path)
    images = [
        str(data_root_directory.joinpath(file))
        for file in samples_df["file_path"].values
    ]
    masks = [
        str(data_root_directory.joinpath(file))
        for file in samples_df["label_path"].values
    ]
    labels: List[int] = samples_df["who_grading"].values.tolist()
    if class_mapping is not None:
        labels = [class_mapping[label] for label in labels]
    return images, masks, labels

from pathlib import Path
from typing import List, Tuple, Optional, Dict

import pandas as pd


def get_samples_df(labels_file: Path) -> pd.DataFrame:
    return pd.read_csv(labels_file, sep="\t")


def get_images_with_labels(
    data_root_directory: Path,
    labels_file_path: Path,
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

from pathlib import Path

import pandas as pd

from meningioma_dl.config import Config


def get_ci_run_samples_df() -> pd.DataFrame:
    return get_samples_df(Config.ci_run_labels_file_path)


def get_training_samples_df() -> pd.DataFrame:
    return get_samples_df(Config.train_labels_file_path)


def get_test_samples_df() -> pd.DataFrame:
    return get_samples_df(Config.test_labels_file_path)


def get_samples_df(labels_file: Path) -> pd.DataFrame:
    labels = pd.read_csv(labels_file, sep="\t")
    assert set(labels.columns) == {"file_path", "label_path", "who_grading"}
    return labels


def get_images_with_labels(data_root_directory, labels_file_path):
    samples_df = get_samples_df(labels_file_path)
    images = [
        str(data_root_directory.joinpath(file))
        for file in samples_df["file_path"].values
    ]
    labels = samples_df["who_grading"].values.tolist()
    return images, labels

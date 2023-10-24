import logging
from pathlib import Path

import fire
from sklearn.model_selection import train_test_split

from meningioma_dl.config import Config

from meningioma_dl.data_loading.labels_loading import get_samples_df


def create_test_split_file(labels_dir: str = "../data/labels"):
    labels_dir = Path(labels_dir)
    labels = get_samples_df(labels_dir.joinpath("labels.tsv"))

    _, test_dataframe = train_test_split(
        labels,
        test_size=Config.test_size,
        random_state=Config.random_seed,
        stratify=labels["who_grading"],
    )
    test_dataframe.to_csv(labels_dir.joinpath("test_labels.tsv"), sep="\t", index=False)


def create_train_validation_split_files(labels_dir: str = "../data/labels"):
    labels_dir = Path(labels_dir)
    all_labels = get_samples_df(labels_dir.joinpath("labels.tsv"))
    test_labels = get_samples_df(labels_dir.joinpath("test_labels.tsv"))

    files_not_in_test = set(all_labels["file_path"]) - set(test_labels["file_path"])
    train_validation_dataframe = all_labels[
        all_labels["file_path"].isin(files_not_in_test)
    ]
    train_dataframe, validation_dataframe = train_test_split(
        train_validation_dataframe,
        test_size=Config.validation_size,
        random_state=Config.random_seed,
        stratify=train_validation_dataframe["who_grading"],
    )
    train_dataframe.to_csv(
        labels_dir.joinpath("train_labels.tsv"), sep="\t", index=False
    )
    validation_dataframe.to_csv(
        labels_dir.joinpath("validation_labels.tsv"), sep="\t", index=False
    )


def create_dataset_split_files(labels_dir: str = "../data/labels"):
    create_test_split_file(labels_dir)
    create_train_validation_split_files(labels_dir)


if __name__ == "__main__":
    try:
        fire.Fire(create_train_validation_split_files)
    except Exception as e:
        logging.error(f"An unexpected error occured {e}", exc_info=True)

import logging
from pathlib import Path

import fire
from sklearn.model_selection import train_test_split, StratifiedKFold

from meningioma_dl.config import Config

from meningioma_dl.data_loading.labels_loading import get_samples_df


def create_test_split_file(labels_dir: str = "../data/labels"):
    labels_dir = Path(labels_dir)
    labels = get_samples_df(labels_dir.joinpath("labels.tsv"))

    _, test_dataframe = train_test_split(
        labels,
        test_size=Config.test_size,
        random_state=123,
        stratify=labels["who_grading"],
    )
    test_dataframe.to_csv(labels_dir.joinpath("test_labels.tsv"), sep="\t", index=False)


def create_train_validation_split_files(
    labels_dir: str = "../data/labels", n_folds: int = 1, random_seed: int = 123
):
    labels_dir = Path(labels_dir)
    all_labels = get_samples_df(labels_dir.joinpath("labels.tsv"))
    test_labels = get_samples_df(labels_dir.joinpath("test_labels.tsv"))

    files_not_in_test = set(all_labels["file_path"]) - set(test_labels["file_path"])
    train_validation_dataframe = all_labels[
        all_labels["file_path"].isin(files_not_in_test)
    ]
    if n_folds == 1:
        train_dataframe, validation_dataframe = train_test_split(
            train_validation_dataframe,
            test_size=Config.validation_size,
            random_state=random_seed,
            stratify=train_validation_dataframe["who_grading"],
        )
        train_dataframes = [train_dataframe]
        validation_dataframes = [validation_dataframe]
    else:
        k_fold = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=random_seed
        )
        train_dataframes = []
        validation_dataframes = []
        for train_index, valid_index in k_fold.split(
            train_validation_dataframe, train_validation_dataframe["who_grading"]
        ):
            train_dataframes.append(train_validation_dataframe.iloc[train_index])
            validation_dataframes.append(train_validation_dataframe.iloc[valid_index])
    for i, (train_dataframe, validation_dataframe) in enumerate(
        zip(train_dataframes, validation_dataframes)
    ):
        train_dataframe.to_csv(
            labels_dir.joinpath(f"train_labels_{i}.tsv"), sep="\t", index=False
        )
        validation_dataframe.to_csv(
            labels_dir.joinpath(f"validation_labels_{i}.tsv"), sep="\t", index=False
        )


def create_dataset_split_files(
    labels_dir: str = "../data/labels", n_folds: int = 1
) -> None:
    # create_test_split_file(labels_dir)
    create_train_validation_split_files(labels_dir)


if __name__ == "__main__":
    try:
        fire.Fire(create_train_validation_split_files)
    except Exception as e:
        logging.error(f"An unexpected error occured {e}", exc_info=True)

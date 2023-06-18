import logging

import fire
from sklearn.model_selection import train_test_split

from meningioma_dl.config import Config

from meningioma_dl.data_loading.labels_loading import get_samples_df


def create_dataset_split_files():
    labels = get_samples_df(Config.labels_file_path)

    train_validation_dataframe, test_dataframe = train_test_split(
        labels,
        test_size=Config.test_size,
        random_state=Config.random_seed,
        stratify=labels["who_grading"],
    )
    train_dataframe, validation_dataframe = train_test_split(
        train_validation_dataframe,
        test_size=Config.validation_size,
        random_state=Config.random_seed,
        stratify=train_validation_dataframe["who_grading"],
    )
    Config.train_labels_file_path.parent.mkdir(parents=True, exist_ok=True)
    Config.validation_labels_file_path.parent.mkdir(parents=True, exist_ok=True)
    Config.test_labels_file_path.parent.mkdir(parents=True, exist_ok=True)
    train_dataframe.to_csv(Config.train_labels_file_path, sep="\t", index=False)
    validation_dataframe.to_csv(
        Config.validation_labels_file_path, sep="\t", index=False
    )
    test_dataframe.to_csv(Config.test_labels_file_path, sep="\t", index=False)


if __name__ == "__main__":
    try:
        fire.Fire(create_dataset_split_files)
    except Exception as e:
        logging.error(f"An unexpected error occured {e}", exc_info=True)

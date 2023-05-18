import logging

import fire
from sklearn.model_selection import train_test_split

from meningioma_dl.config import Configuration
import pandas as pd


def create_dataset_split_files():
    labels = pd.read_csv(Configuration.labels_file_path, sep="\t")

    train_dataframe, test_dataframe = train_test_split(
        labels,
        test_size=Configuration.test_size,
        random_state=Configuration.random_seed,
        stratify=labels["who_grading"],
    )
    Configuration.train_labels_file_path.parent.mkdir(parents=True, exist_ok=True)
    Configuration.test_labels_file_path.parent.mkdir(parents=True, exist_ok=True)
    train_dataframe.to_csv(Configuration.train_labels_file_path, sep="\t", index=False)
    test_dataframe.to_csv(Configuration.test_labels_file_path, sep="\t", index=False)


if __name__ == "__main__":
    try:
        fire.Fire(create_dataset_split_files)
    except Exception as e:
        logging.error(f"An unexpected error occured {e}", exc_info=True)

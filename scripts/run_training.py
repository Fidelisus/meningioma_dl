import logging

import fire as fire
import optuna

from meningioma_dl.config import Config
from meningioma_dl.data_loading.labels_loading import get_training_samples_df


def main():
    labels_df = get_training_samples_df()

    study = optuna.create_study(
        study_name=study_name, storage=Config.optuna_database_directory
    )


if __name__ == "__main__":
    try:
        fire.Fire(main)
    except Exception as e:
        logging.error(f"An unexpected error occured {e}", exc_info=True)

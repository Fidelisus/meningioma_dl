import logging
from typing import Any, Optional, Dict

import fire
import optuna
from optuna import Trial

from meningioma_dl.config import Config
from meningioma_dl.data_loading.augmentation import (
    propose_augmentation,
    suggest_hyperparameter_value,
)
from meningioma_dl.evaluate import evaluate
from meningioma_dl.hyperparameters_configs import SEARCH_SPACES
from meningioma_dl.train import train
from meningioma_dl.utils import generate_run_id, setup_logging


test_run_params: Dict[str, Any] = {"learning_rate": (0.01, 0.1)}

#  optuna-dashboard sqlite:///C:\Users\Lenovo\Desktop\meningioma_project\meningioma_dl\data\optuna\optuna_store.db


def suggest_parameters_values(
    trial: Trial, augmentation_settings: Dict[str, Any]
) -> dict[str, Any]:
    parameters_values: Dict[str, Any] = {}
    for name, values in augmentation_settings.items():
        parameters_values[name] = suggest_hyperparameter_value(
            trial, name, values, float
        )
    return parameters_values


def run_study(
    env_file_path: str,
    n_trials: int = 10,
    n_epochs: int = 10,
    study_name: str = "some_run",
    run_id: Optional[str] = None,
    device_name: str = "cpu",
    n_workers: int = 1,
    search_space_name: str = "full",
):
    search_space = SEARCH_SPACES[search_space_name]

    def objective(trial):
        transforms = propose_augmentation(trial, search_space)
        _, trained_model_path = train(
            env_file_path=None,
            run_id=run_id,
            augmentation_settings=transforms,
            n_epochs=n_epochs,
            device_name=device_name,
            n_workers=n_workers,
            **suggest_parameters_values(trial, test_run_params),
        )
        if trained_model_path is None:
            raise ValueError("No model was created during training, aborting.")

        best_f_score = evaluate(
            trained_model_path=trained_model_path, device_name=device_name
        )
        return best_f_score

    if run_id is None:
        run_id = f"{study_name}_{generate_run_id()}"
    else:
        run_id = f"{study_name}_{run_id}"
    Config.load_env_variables(env_file_path, run_id)
    setup_logging(Config.log_file_path)

    logging.info("The following search space will be used:")
    logging.info(search_space)

    study = optuna.create_study(
        storage=Config.optuna_database_directory, study_name=run_id
    )
    study.optimize(objective, n_trials=n_trials)


if __name__ == "__main__":
    try:
        fire.Fire(run_study)
    except Exception as e:
        logging.error(f"An unexpected error occured {e}", exc_info=True)
        raise

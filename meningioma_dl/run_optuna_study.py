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
from meningioma_dl.hyperparameters_configs import SEARCH_SPACES, HYPERPARAMETERS_CONFIGS
from meningioma_dl.train import train
from meningioma_dl.utils import generate_run_id, setup_logging

#  optuna-dashboard sqlite:///C:\Users\Lenovo\Desktop\meningioma_project\meningioma_dl\data\optuna\optuna_store.db


def suggest_parameters_values(
    trial: Trial, augmentation_settings: Dict[str, Any]
) -> Dict[str, Any]:
    parameters_values: Dict[str, Any] = {}
    for name, values in augmentation_settings.items():
        parameters_values[name] = suggest_hyperparameter_value(
            trial, name, values, float
        )
    return parameters_values


def run_study(
    env_file_path: str,
    hyperparameters_config_name: str,
    n_trials: int = 1,
    n_epochs: int = 1,
    study_name: str = "some_run",
    run_id: Optional[str] = None,
    device_name: str = "cpu",
    n_workers: int = 1,
    search_space_name: str = "affine_transforms",
    batch_size:int = 2,
    validation_interval: int = 1,
    save_model:bool=False,
):
    search_space = SEARCH_SPACES[search_space_name]
    hyperparameters_config = HYPERPARAMETERS_CONFIGS[hyperparameters_config_name]

    def objective(trial: Trial):
        transforms = propose_augmentation(trial, search_space)
        hyperparameters_values = suggest_parameters_values(trial, hyperparameters_config)
        logging.info(f"Transforms: {transforms}")
        logging.info(f"Hyperparameters: {hyperparameters_values}")
        _, trained_model_path = train(
            env_file_path=None,
            run_id=run_id,
            augmentation_settings=transforms,
            n_epochs=n_epochs,
            device_name=device_name,
            n_workers=n_workers,
            batch_size=batch_size,
            validation_interval=validation_interval,
            trial_id=trial.number,
            save_model=save_model,
            **hyperparameters_values,
        )
        if trained_model_path is None:
            raise ValueError("No model was created during training, aborting.")

        best_f_score = evaluate(
            trained_model_path=trained_model_path,
            device_name=device_name,
            visualizations_folder=Config.visualizations_directory.joinpath(run_id, "evaluation")
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
        storage=Config.optuna_database_directory, study_name=run_id, direction="maximize"
    )
    study.optimize(objective, n_trials=n_trials)


if __name__ == "__main__":
    try:
        fire.Fire(run_study)
    except Exception as e:
        logging.error(f"An unexpected error occured {e}", exc_info=True)
        raise

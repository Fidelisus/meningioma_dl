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
from meningioma_dl.experiments_configs.experiments import (
    AUGMENTATIONS_SEARCH_SPACES,
    HYPERPARAMETERS_CONFIGS,
    PREPROCESSING_SETTINGS,
)
from meningioma_dl.train import train
from meningioma_dl.training_utils import SCHEDULERS
from meningioma_dl.utils import generate_run_id, setup_logging

#  optuna-dashboard sqlite:///C:\Users\Lenovo\Desktop\meningioma_project\meningioma_dl\data\optuna\optuna_store.db


def suggest_parameters_values(
    trial: Trial, augmentation_settings: Dict[str, Any]
) -> Dict[str, Any]:
    parameters_values: Dict[str, Any] = {}
    for name, values in augmentation_settings.items():
        if isinstance(values[1], int):
            parameters_values[name] = suggest_hyperparameter_value(
                trial, name, values, int
            )
        else:
            parameters_values[name] = suggest_hyperparameter_value(
                trial, name, values, float
            )
    return parameters_values


def run_study(
    env_file_path: str,
    hyperparameters_config_name: str = "simple_conf_exp_1",
    n_trials: int = 1,
    n_epochs: int = 1,
    study_name: str = "some_run",
    run_id: Optional[str] = None,
    device_name: str = "cpu",
    n_workers: int = 1,
    search_space_name: str = "affine_transforms",
    batch_size: int = 2,
    validation_interval: int = 1,
    save_intermediate_models: bool = False,
    scheduler_name: str = "exponent",
    preprocessing_settings_name: str = "default",
):
    def objective(trial: Trial):
        transforms = propose_augmentation(trial, search_space)
        hyperparameters_values = suggest_parameters_values(
            trial, hyperparameters_config
        )
        logging.info(f"Transforms: {transforms}")
        logging.info(f"Hyperparameters: {hyperparameters_values}")
        visualizations_folder = Config.visualizations_directory.joinpath(
            run_id, str(trial.number)
        )
        scheduler = SCHEDULERS[scheduler_name]
        _, trained_model_path = train(
            env_file_path=None,
            run_id=run_id,
            augmentation_settings=transforms,
            n_epochs=n_epochs,
            device_name=device_name,
            n_workers=n_workers,
            batch_size=batch_size,
            validation_interval=validation_interval,
            visualizations_folder=visualizations_folder,
            save_intermediate_models=save_intermediate_models,
            saved_models_folder=Config.saved_models_directory.joinpath(
                run_id, str(trial.number)
            ),
            scheduler=scheduler,
            learning_rate=hyperparameters_values.pop("learning_rate"),
            scheduler_parameters=hyperparameters_values,
            preprocessing_settings=preprocessing_settings,
        )
        if trained_model_path is None:
            raise ValueError("No model was created during training, aborting.")

        f_score_of_the_best_model = evaluate(
            trained_model_path=trained_model_path,
            device_name=device_name,
            visualizations_folder=visualizations_folder,
            batch_size=batch_size,
            preprocessing_settings=preprocessing_settings,
        )
        return f_score_of_the_best_model

    if run_id is None:
        run_id = f"{study_name}_{generate_run_id()}"
    else:
        run_id = f"{study_name}_{run_id}"
    Config.load_env_variables(env_file_path, run_id)
    setup_logging(Config.log_file_path)

    search_space = AUGMENTATIONS_SEARCH_SPACES[search_space_name]
    hyperparameters_config = HYPERPARAMETERS_CONFIGS[hyperparameters_config_name]
    preprocessing_settings = PREPROCESSING_SETTINGS[preprocessing_settings_name]
    logging.info(f"run_id: {run_id}")
    logging.info(
        f"hyperparameters_config_name: {hyperparameters_config_name}, search_space_name: {search_space_name}"
    )
    logging.info(f"Augmentations search space: {search_space}")
    logging.info(f"Hyperparameters search space: {hyperparameters_config}")
    logging.info(f"preprocessing_settings: {preprocessing_settings}")
    logging.info(
        f"n_epochs: {n_epochs}, n_trials: {n_trials}, "
        f"batch_size: {batch_size}, validation_interval: {validation_interval}"
    )

    optuna.logging.enable_propagation()
    study = optuna.create_study(
        storage=Config.optuna_database_directory,
        study_name=run_id,
        direction="maximize",
    )
    study.optimize(objective, n_trials=n_trials)


if __name__ == "__main__":
    try:
        fire.Fire(run_study)
    except Exception as e:
        logging.error(f"An unexpected error occured {e}", exc_info=True)
        raise

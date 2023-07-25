import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import optuna
import fire
import shortuuid
from optuna import Trial

from meningioma_dl.config import Config
from meningioma_dl.data_loading.augmentation import (
    create_augmentation,
    _suggest_hyperparameter_value,
)
from meningioma_dl.evaluate import evaluate
from meningioma_dl.train import train

test_run_augment: dict[str, dict[str, tuple]] = {
    "rand_flip": {"spatial_axis": (0, 1), "prob": 0.5},
    "rand_rotate": {"prob": (0, 1)},
}

test_run_params: dict[str, Any] = {"learning_rate": (0.01, 0.1)}

#  optuna-dashboard sqlite:///C:\Users\Lenovo\Desktop\meningioma_project\meningioma_dl\data\optuna\optuna_store.db


def suggest_parameters_values(
    trial: Trial, augmentation_settings: dict[str, Any]
) -> dict[str, Any]:
    parameters_values: dict[str, Any] = {}
    for name, values in augmentation_settings.items():
        parameters_values[name] = _suggest_hyperparameter_value(
            trial, name, values, float
        )
    return parameters_values


def run_study(
    n_trials: int = 2, n_epochs: int = 1, trial_name: str = "more_augmentations"
):
    def objective(trial):
        transforms = create_augmentation(trial, augmentation_settings)
        _, trained_model_path = train(
            augmentation_settings=transforms,
            n_epochs=n_epochs,
            **suggest_parameters_values(trial, test_run_params),
        )
        if trained_model_path is None:
            raise ValueError("No model was created by training, aborting.")

        best_f_score = evaluate(trained_model_path=Path(trained_model_path))
        return best_f_score

    study_id = f"{trial_name}_{datetime.now().isoformat()}_{shortuuid.uuid()}"
    augmentation_settings = test_run_augment
    study = optuna.create_study(
        storage=Config.optuna_database_directory, study_name=study_id
    )
    print(augmentation_settings)
    study.optimize(objective, n_trials=n_trials)


if __name__ == "__main__":
    try:
        fire.Fire(run_study)
    except Exception as e:
        logging.error(f"An unexpected error occured {e}", exc_info=True)
        raise

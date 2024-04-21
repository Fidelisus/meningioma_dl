import logging
from typing import Optional

import fire
import optuna
from optuna import Trial

from meningioma_dl.config import Config
from meningioma_dl.evaluate import evaluate
from meningioma_dl.experiments_specs.augmentation_specs import AugmentationSpecs
from meningioma_dl.experiments_specs.model_specs import ModelSpecs
from meningioma_dl.experiments_specs.modelling_specs import (
    ModellingSpecs,
)
from meningioma_dl.experiments_specs.preprocessing_specs import (
    PreprocessingSpecs,
)
from meningioma_dl.experiments_specs.scheduler_specs import SchedulerSpecs
from meningioma_dl.experiments_specs.training_specs import (
    get_training_specs,
    CentralizedTrainingSpecs,
)
from meningioma_dl.train import train
from meningioma_dl.utils import generate_run_id, setup_logging


#  optuna-dashboard sqlite:///C:\Users\Lenovo\Desktop\meningioma_project\meningioma_dl\data\optuna\optuna_store.db


def run_study(
    env_file_path: str,
    n_trials: int = 1,
    run_id: Optional[str] = None,
    device_name: str = "cpu",
    validation_interval: int = 1,
    preprocessing_specs_name: str = "no_resize",
    augmentations_specs_name: str = "basic_01p",
    scheduler_specs_name: str = "05_lr_099_gamma",
    model_specs_name: str = "resnet_10_2_unfreezed",
    training_specs_name: str = "central_1_epochs",
):
    def objective(trial: Trial):
        visualizations_folder = Config.visualizations_directory.joinpath(run_id)
        _, trained_model_path = train(
            env_file_path=None,
            run_id=run_id,
            device_name=device_name,
            validation_interval=validation_interval,
            visualizations_folder=visualizations_folder,
            saved_models_folder=Config.saved_models_directory.joinpath(run_id),
            modelling_specs=modelling_spec,
            training_specs=training_spec,
        )
        if trained_model_path is None:
            raise ValueError("No model was created during training, aborting.")

        f_score_of_the_best_model = evaluate(
            run_id=run_id,
            trained_model_path=trained_model_path,
            device_name=device_name,
            visualizations_folder=visualizations_folder,
            model_specs=modelling_spec.model_specs,
            preprocessing_specs=modelling_spec.preprocessing_specs,
            training_specs=training_spec,
        )
        return f_score_of_the_best_model

    if run_id is None:
        run_id = generate_run_id()

    Config.load_env_variables(env_file_path, run_id)
    setup_logging(Config.log_file_path)

    modelling_spec = ModellingSpecs(
        PreprocessingSpecs.get_from_name(preprocessing_specs_name),
        AugmentationSpecs.get_from_name(augmentations_specs_name),
        SchedulerSpecs.get_from_name(scheduler_specs_name),
        ModelSpecs.get_from_name(model_specs_name),
    )
    training_spec: CentralizedTrainingSpecs = get_training_specs(training_specs_name)
    logging.info(f"run_id: {run_id}")
    logging.info(f"Modelling specs: {modelling_spec}")
    logging.info(f"Augmentations specs name: {augmentations_specs_name}")
    logging.info(f"Training specs: {training_spec}")
    logging.info(f"n_trials: {n_trials}, validation_interval: {validation_interval}")

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

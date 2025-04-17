import logging
from pathlib import Path
from typing import Literal

import fire

from meningioma_dl.config import Config
from meningioma_dl.experiments_specs.model_specs import ModelSpecs
from meningioma_dl.experiments_specs.preprocessing_specs import PreprocessingSpecs
from meningioma_dl.experiments_specs.training_specs import (
    get_training_specs,
)
from meningioma_dl.model_evaluation.centralized_evaluation import centralized_evaluation
from meningioma_dl.jobs.experiments_setup import setup_run, generate_run_id


def run_standalone_evaluate(
    trained_model_path: str,
    env_file_path: str,
    run_id: str = generate_run_id(),
    device_name: Literal["cpu", "cuda"] = "cpu",
    preprocessing_specs_name: str = "no_resize",
    model_specs_name: str = "resnet_10_0_unfreezed",
    use_test_data: bool = False,
    cv_fold: int = None,
    manual_seed: int = 123,
):
    config = setup_run(
        env_file_path=env_file_path,
        manual_seed=manual_seed,
        device_name=device_name,
        cv_fold=cv_fold,
    )

    labels_file = (
        config.test_labels_file_path
        if use_test_data
        else config.validation_labels_file_path
    )
    visualizations_folder = config.visualizations_directory.joinpath(run_id)
    centralized_evaluation(
        trained_model_path=Path(trained_model_path),
        device=config.device,
        config=config,
        visualizations_folder=visualizations_folder,
        model_specs=ModelSpecs.get_from_name(model_specs_name),
        preprocessing_specs=PreprocessingSpecs.get_from_name(preprocessing_specs_name),
        training_specs=get_training_specs("evaluation"),
        test_data_path=labels_file,
    )


if __name__ == "__main__":
    try:
        fire.Fire(run_standalone_evaluate)
    except Exception as e:
        logging.error(f"An unexpected error occured {e}", exc_info=True)

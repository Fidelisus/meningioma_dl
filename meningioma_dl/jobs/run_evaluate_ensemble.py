import logging
from typing import Optional, Literal

import fire
import torch

from data_loading.experiments_setup import generate_run_id
from localized_federated_ensemble.evaluation import (
    evaluate_globally_weighted_ensemble_model,
    evaluate_locally_weighted_ensemble_model,
)
from localized_federated_ensemble.weights_calculation import EnsembleWeighting
from meningioma_dl.data_loading.experiments_setup import (
    setup_run,
)
from meningioma_dl.experiments_specs.model_specs import ModelSpecs
from meningioma_dl.experiments_specs.preprocessing_specs import PreprocessingSpecs
from meningioma_dl.experiments_specs.training_specs import (
    get_training_specs,
)
from meningioma_dl.visualizations.results_visualizations import (
    create_evaluation_report,
)


def run_standalone_ensemble_evaluate(
    env_file_path: str,
    weights_folder: str,
    model_id: str,
    run_id: str = generate_run_id(),
    device_name: Literal["cpu", "cuda"] = "cpu",
    preprocessing_specs_name: str = "no_resize",
    model_specs_name: str = "class_2_and_3_together_4_unfreezed",
    manual_seed: int = 123,
    ensemble_weighting: str = "majority_voting",
    training_specs_name: str = "federated_local_run",
):
    config = setup_run(
        env_file_path=env_file_path,
        manual_seed=manual_seed,
        device_name=device_name,
    )
    ensemble_weighting = EnsembleWeighting(ensemble_weighting)
    model_specs = ModelSpecs.get_from_name(model_specs_name)
    training_specs = get_training_specs(training_specs_name)
    weights_folder = config.visualizations_directory.joinpath(weights_folder)

    trained_model_paths = []
    for client_id in range(training_specs.number_of_clients):
        trained_model_paths.append(
            str(
                config.saved_models_directory.joinpath(
                    model_id, f"best_model_{client_id}.pth.tar"
                )
            )
        )
    logging.info(
        f"Starting ensemble model evaluation using the models from {trained_model_paths}"
        f" and ensemble_weighting {ensemble_weighting}"
    )
    torch.manual_seed(manual_seed)
    logging.info(f"Samples to be used are read from {config.test_labels_file_path}")

    if ensemble_weighting is EnsembleWeighting.GLOBAL_WEIGHTING:
        logging.info(f"Using global model weighting with weights from {weights_folder}")
        validation_metrics = evaluate_globally_weighted_ensemble_model(
            trained_model_paths=trained_model_paths,
            device=config.device,
            model_specs=ModelSpecs.get_from_name(model_specs_name),
            preprocessing_specs=PreprocessingSpecs.get_from_name(
                preprocessing_specs_name
            ),
            test_data_path=config.test_labels_file_path,
            weights_folder=weights_folder,
            data_directory=config.data_directory,
        )
    elif ensemble_weighting is EnsembleWeighting.LOCAL_WEIGHTING:
        validation_metrics = evaluate_locally_weighted_ensemble_model(
            trained_model_paths=trained_model_paths,
            device=config.device,
            weights_folder=weights_folder,
            model_specs=model_specs,
            preprocessing_specs=PreprocessingSpecs.get_from_name(
                preprocessing_specs_name
            ),
            training_specs=training_specs,
            test_data_path=config.test_labels_file_path,
            data_directory=config.data_directory,
        )
    else:
        raise NotImplementedError(f"{ensemble_weighting=} is not implemented yet")
    create_evaluation_report(
        true=validation_metrics.true,
        predictions=validation_metrics.predictions,
        visualizations_folder=config.visualizations_directory.joinpath(run_id),
        run_id=run_id,
        model_specs=model_specs,
        training_specs=training_specs,
        n_classes=model_specs.number_of_classes,
    )


if __name__ == "__main__":
    try:
        fire.Fire(run_standalone_ensemble_evaluate)
    except Exception as e:
        logging.error(f"An unexpected error occured {e}", exc_info=True)

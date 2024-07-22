import json
import logging
from enum import Enum
from pathlib import Path
from typing import Optional, Callable, Tuple, List, Dict

import fire
import numpy as np
import torch
from torch.utils.data import DataLoader

from meningioma_dl.config import Config
from meningioma_dl.data_loading.data_loader import (
    get_data_loader,
    TransformationsMode,
)
from meningioma_dl.evaluate import calculate_basic_metrics, load_model_from_file
from meningioma_dl.experiments_specs.model_specs import ModelSpecs
from meningioma_dl.experiments_specs.preprocessing_specs import PreprocessingSpecs
from meningioma_dl.experiments_specs.training_specs import (
    CentralizedTrainingSpecs,
    get_training_specs,
    FederatedTrainingSpecs,
)
from meningioma_dl.federated_learning.ensemble_fl import ensemble_weights_to_numpy
from meningioma_dl.federated_learning.federated_training_utils import (
    get_federated_data_loaders,
)
from meningioma_dl.models.resnet import ResNet
from meningioma_dl.training_utils import (
    get_model_predictions,
)
from meningioma_dl.utils import (
    setup_run,
)
from meningioma_dl.visualizations.results_visualizations import (
    create_evaluation_report,
    ValidationMetrics,
    merge_validation_metrics_true_and_pred,
)

AVAILABLE_ENSEMBLES = {
    "centralized_2_classes": (
        (
            "cv_final_model_centralized_central_300_epochs_no_resize_with_bias_correction_1p_001_lr_099_gamma_class_2_and_3_together_4_unfreezed_fold0_6376843",
            -1,
        ),
        (
            "cv_final_model_centralized_central_300_epochs_no_resize_with_bias_correction_1p_001_lr_099_gamma_class_2_and_3_together_4_unfreezed_fold1_6376844",
            -1,
        ),
        (
            "cv_final_model_centralized_central_300_epochs_no_resize_with_bias_correction_1p_001_lr_099_gamma_class_2_and_3_together_4_unfreezed_fold2_6376845",
            -1,
        ),
        (
            "cv_final_model_centralized_central_300_epochs_no_resize_with_bias_correction_1p_001_lr_099_gamma_class_2_and_3_together_4_unfreezed_fold3_6376846",
            -1,
        ),
        (
            "cv_final_model_centralized_central_300_epochs_no_resize_with_bias_correction_1p_001_lr_099_gamma_class_2_and_3_together_4_unfreezed_fold4_6376847",
            -1,
        ),
    ),
    "centralized_3_classes": (
        (
            "cv_final_model_centralized_central_300_epochs_no_resize_with_bias_correction_08p_0001_lr_099_gamma_resnet_10_4_unfreezed_fold0_6376848",
            -1,
        ),
        (
            "cv_final_model_centralized_central_300_epochs_no_resize_with_bias_correction_08p_0001_lr_099_gamma_resnet_10_4_unfreezed_fold1_6376849",
            -1,
        ),
        (
            "cv_final_model_centralized_central_300_epochs_no_resize_with_bias_correction_08p_0001_lr_099_gamma_resnet_10_4_unfreezed_fold2_6384955",
            -1,
        ),
        (
            "cv_final_model_centralized_central_300_epochs_no_resize_with_bias_correction_08p_0001_lr_099_gamma_resnet_10_4_unfreezed_fold3_6384956",
            -1,
        ),
        (
            "cv_final_model_centralized_central_300_epochs_no_resize_with_bias_correction_08p_0001_lr_099_gamma_resnet_10_4_unfreezed_fold4_6384957",
            -1,
        ),
    ),
    "local": (
        (
            "cv_final_model_centralized_central_300_epochs_no_resize_with_bias_correction_1p_001_lr_099_gamma_class_2_and_3_together_4_unfreezed_fold0_6376843",
            -1,
        ),
        (
            "cv_final_model_centralized_central_300_epochs_no_resize_with_bias_correction_1p_001_lr_099_gamma_class_2_and_3_together_4_unfreezed_fold1_6376844",
            -1,
        ),
        (
            "cv_final_model_centralized_central_300_epochs_no_resize_with_bias_correction_1p_001_lr_099_gamma_class_2_and_3_together_4_unfreezed_fold2_6376845",
            -1,
        ),
    ),
    "fl_local_weighting_local": (
        ("19-07-24_17-37-58_RSg3iQtVQvRCYQ35YgpDcV", 0),
        ("19-07-24_17-37-58_RSg3iQtVQvRCYQ35YgpDcV", 1),
    ),
}


class EnsembleWeighting(Enum):
    MAJORITY_VOTING = "majority_voting"
    GLOBAL_WEIGHTING = "global_weighting"
    LOCAL_WEIGHTING = "local_weighting"


def evaluate_ensemble_model(
    trained_model_paths: List[str],
    test_data_path: Path,
    device: torch.device = torch.device("cpu"),
    model_specs: ModelSpecs = ModelSpecs(),
    preprocessing_specs: PreprocessingSpecs = PreprocessingSpecs(),
    training_specs: CentralizedTrainingSpecs = CentralizedTrainingSpecs(),
    logger: Callable[[str], None] = logging.info,
    weights_folder: Optional[Path] = None,
) -> ValidationMetrics:
    data_loader, labels = get_data_loader(
        test_data_path,
        Config.data_directory,
        transformations_mode=TransformationsMode.ONLY_PREPROCESSING,
        batch_size=training_specs.batch_size,
        preprocessing_specs=preprocessing_specs,
        class_mapping=model_specs.class_mapping,
        client_specific_preprocessing=training_specs.client_specific_preprocessing,
    )

    models = load_models(trained_model_paths, model_specs)

    if weights_folder is None:
        ensemble_models_weights = None
    elif weights_folder:
        ensemble_models_weights = load_json(
            weights_folder, "global_ensemble_weights.json"
        )
        ensemble_models_weights = ensemble_weights_to_numpy(ensemble_models_weights)
    else:
        raise NotImplemented(f"Ensemble weighting {ensemble_weighting} not implemented")

    f_score, validation_metrics = evaluate_ensemble(
        data_loader,
        models,
        model_specs,
        device,
        logger,
        ensemble_models_weights=ensemble_models_weights,
    )

    return validation_metrics


def evaluate_ensemble_model_local_weighting(
    trained_model_paths: List[str],
    test_data_path: Path,
    weights_folder: Path,
    device: torch.device = torch.device("cpu"),
    model_specs: ModelSpecs = ModelSpecs(),
    preprocessing_specs: PreprocessingSpecs = PreprocessingSpecs(),
    training_specs: FederatedTrainingSpecs = FederatedTrainingSpecs(),
    logger: Callable[[str], None] = logging.info,
) -> ValidationMetrics:
    test_data_loaders, labels_test = get_federated_data_loaders(
        labels_file_path=test_data_path,
        data_root_directory=Config.data_directory,
        transformations_mode=TransformationsMode.ONLY_PREPROCESSING,
        training_specs=training_specs,
        default_preprocessing_specs=preprocessing_specs,
        class_mapping=model_specs.class_mapping,
    )
    logger(f"Validation data loaded from {test_data_loaders}")

    validation_metrics_all_clients = []
    for client_id, data_loader in test_data_loaders.items():
        models = load_models(trained_model_paths, model_specs)

        # TODO TODO pass the location
        ensemble_models_weights = load_json(
            weights_folder, f"local_ensemble_weights_{client_id}.json"
        )
        ensemble_models_weights = ensemble_weights_to_numpy(ensemble_models_weights)

        f_score, validation_metrics = evaluate_ensemble(
            data_loader,
            models,
            model_specs,
            device,
            logger,
            ensemble_models_weights=ensemble_models_weights,
        )
        validation_metrics_all_clients.append(validation_metrics)

    validation_metrics_aggregated = merge_validation_metrics_true_and_pred(
        validation_metrics_all_clients
    )
    return validation_metrics_aggregated


def load_models(trained_model_paths: List[str], model_specs: ModelSpecs):
    models: List[ResNet] = []
    for trained_model_path in trained_model_paths:
        model = load_model_from_file(
            trained_model_path, model_specs, torch.device("cpu")
        )
        models.append(model)
    return models


def load_json(weights_folder: Path, file_name: str) -> Dict:
    with open(
        weights_folder.joinpath(file_name),
        "r",
    ) as f:
        return json.load(f)


def evaluate_ensemble(
    data_loader: DataLoader,
    models: List[ResNet],
    model_specs: ModelSpecs,
    device: torch.device,
    logger: Callable[[str], None] = logging.info,
    ensemble_models_weights: Optional[np.ndarray] = None,
) -> Tuple[float, ValidationMetrics]:
    if not models:
        raise ValueError("More than one model is needed to evaluate an ensemble.")

    all_predictions = {}
    for i, model in enumerate(models):
        model.to(device)
        model.eval()
        with torch.no_grad():
            labels, predictions, images_paths = get_model_predictions(
                data_loader, model, device
            )
        model.cpu()
        labels_cpu = labels.cpu()
        all_predictions[i] = predictions.cpu()

    predictions_matrix = torch.stack(list(all_predictions.values()))
    if ensemble_models_weights is not None:
        predictions_matrix = predictions_matrix * ensemble_models_weights[:, None, None]

    predictions_flat = predictions_matrix.argmax(dim=2).mode(dim=0).values
    f_score = calculate_basic_metrics(
        labels_cpu, predictions_flat, model_specs.evaluation_metric_weighting, logger
    )
    return f_score, ValidationMetrics(f_score, None, labels_cpu, predictions_flat)


def run_ensemble_evaluate(
    ensemble_id: str,
    env_file_path: Optional[str] = None,
    run_id: Optional[str] = None,
    device_name: str = "cpu",
    preprocessing_specs_name: str = "no_resize",
    model_specs_name: str = "class_2_and_3_together_4_unfreezed",
    manual_seed: int = 123,
    ensemble_weighting: str = "majority_voting",
    weights_folder: Optional[str] = None,
):
    device, run_id = setup_run(env_file_path, run_id, manual_seed, device_name, None)
    ensemble_weighting = EnsembleWeighting(ensemble_weighting)
    visualizations_folder = Config.visualizations_directory.joinpath(run_id)
    model_specs = ModelSpecs.get_from_name(model_specs_name)
    # TODO make it nicer
    training_specs = get_training_specs("federated_1r_1e_2c")

    if (
        ensemble_weighting is EnsembleWeighting.MAJORITY_VOTING
        and weights_folder is not None
    ):
        raise ValueError(
            f"You cannot pass weights_folder if ensemble_weighting is MAJORITY_VOTING"
        )
    if weights_folder is not None:
        weights_folder = Config.visualizations_directory.joinpath(weights_folder)

    trained_model_paths = []
    for model_id, client_id in AVAILABLE_ENSEMBLES[ensemble_id]:
        trained_model_paths.append(
            str(
                Config.saved_models_directory.joinpath(
                    model_id, f"epoch_{client_id}.pth.tar"
                )
            )
        )
    logging.info(
        f"Starting ensemble model evaluation using the models from {trained_model_paths}"
        f" and ensemble_weighting {ensemble_weighting}"
    )
    torch.manual_seed(manual_seed)
    logging.info(f"Samples to be used are read from {Config.test_labels_file_path}")

    if ensemble_weighting is not EnsembleWeighting.LOCAL_WEIGHTING:
        validation_metrics = evaluate_ensemble_model(
            trained_model_paths=trained_model_paths,
            device=device,
            model_specs=ModelSpecs.get_from_name(model_specs_name),
            preprocessing_specs=PreprocessingSpecs.get_from_name(
                preprocessing_specs_name
            ),
            training_specs=get_training_specs("evaluation"),
            test_data_path=Config.test_labels_file_path,
            weights_folder=weights_folder,
        )
    else:
        validation_metrics = evaluate_ensemble_model_local_weighting(
            trained_model_paths=trained_model_paths,
            device=device,
            weights_folder=weights_folder,
            model_specs=model_specs,
            preprocessing_specs=PreprocessingSpecs.get_from_name(
                preprocessing_specs_name
            ),
            training_specs=training_specs,
            test_data_path=Config.test_labels_file_path,
        )
    create_evaluation_report(
        validation_metrics.true,
        validation_metrics.predictions,
        visualizations_folder,
        run_id,
        model_specs,
        training_specs,
        model_specs.number_of_classes,
    )


if __name__ == "__main__":
    try:
        fire.Fire(run_ensemble_evaluate)
    except Exception as e:
        logging.error(f"An unexpected error occured {e}", exc_info=True)

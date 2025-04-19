import logging
from pathlib import Path
from typing import List, Optional, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from meningioma_dl.data_loading.data_loader import get_data_loader, TransformationsMode
from meningioma_dl.experiments_specs.model_specs import ModelSpecs
from meningioma_dl.experiments_specs.preprocessing_specs import PreprocessingSpecs
from meningioma_dl.experiments_specs.training_specs import FederatedTrainingSpecs
from meningioma_dl.federated_learning.data_loading import get_federated_data_loaders
from meningioma_dl.localized_federated_ensemble.data_loading import (
    load_models,
    load_json,
)
from meningioma_dl.localized_federated_ensemble.weights_calculation import (
    ensemble_weights_to_numpy,
)
from meningioma_dl.model_evaluation.metrics import calculate_basic_metrics
from meningioma_dl.model_training.predictions import get_model_predictions
from meningioma_dl.models.resnet import ResNet
from meningioma_dl.visualizations.results_visualizations import (
    ValidationMetrics,
    merge_validation_metrics_true_and_pred,
)


def evaluate_globally_weighted_ensemble_model(
    trained_model_paths: List[str],
    test_data_path: Path,
    weights_folder: Path,
    data_directory: Path,
    device: torch.device = torch.device("cpu"),
    model_specs: ModelSpecs = ModelSpecs(),
    preprocessing_specs: PreprocessingSpecs = PreprocessingSpecs(),
    client_specific_preprocessing: Optional[dict[int, Optional[str]]] = None,
    logger: Callable[[str], None] = logging.info,
) -> ValidationMetrics:
    data_loader, labels = get_data_loader(
        labels_file_path=test_data_path,
        data_root_directory=data_directory,
        transformations_mode=TransformationsMode.ONLY_PREPROCESSING,
        preprocessing_specs=preprocessing_specs,
        class_mapping=model_specs.class_mapping,
        client_specific_preprocessing=client_specific_preprocessing,
    )
    logger(f"Test data loaded from {data_loader}")
    models = load_models(trained_model_paths, model_specs)
    ensemble_models_weights = ensemble_weights_to_numpy(
        load_json(weights_folder, "global_ensemble_weights.json")
    )
    return ensemble_evaluation_loop(
        data_loader=data_loader,
        models=models,
        model_specs=model_specs,
        device=device,
        logger=logger,
        ensemble_models_weights=ensemble_models_weights,
    )


def evaluate_locally_weighted_ensemble_model(
    trained_model_paths: List[str],
    test_data_path: Path,
    weights_folder: Path,
    data_directory: Path,
    device: torch.device = torch.device("cpu"),
    model_specs: ModelSpecs = ModelSpecs(),
    preprocessing_specs: PreprocessingSpecs = PreprocessingSpecs(),
    training_specs: FederatedTrainingSpecs = FederatedTrainingSpecs(),
    logger: Callable[[str], None] = logging.info,
) -> ValidationMetrics:
    test_data_loaders, labels_test = get_federated_data_loaders(
        labels_file_path=test_data_path,
        data_root_directory=data_directory,
        transformations_mode=TransformationsMode.ONLY_PREPROCESSING,
        training_specs=training_specs,
        default_preprocessing_specs=preprocessing_specs,
        class_mapping=model_specs.class_mapping,
    )
    logger(f"Test data loaded from {test_data_loaders}")

    validation_metrics_all_clients = []
    for client_id, data_loader in test_data_loaders.items():
        models = load_models(trained_model_paths, model_specs)
        ensemble_models_weights = ensemble_weights_to_numpy(
            load_json(weights_folder, f"local_ensemble_weights_{client_id}.json")
        )
        validation_metrics = ensemble_evaluation_loop(
            data_loader=data_loader,
            models=models,
            model_specs=model_specs,
            device=device,
            logger=logger,
            ensemble_models_weights=ensemble_models_weights,
        )
        validation_metrics_all_clients.append(validation_metrics)
    return merge_validation_metrics_true_and_pred(validation_metrics_all_clients)


def ensemble_evaluation_loop(
    data_loader: DataLoader,
    models: List[ResNet],
    model_specs: ModelSpecs,
    device: torch.device,
    logger: Callable[[str], None] = logging.info,
    ensemble_models_weights: Optional[np.ndarray] = None,
) -> ValidationMetrics:
    if not models:
        raise ValueError("More than one model is needed to evaluate an ensemble.")

    all_predictions = {}
    labels = None
    for i, model in enumerate(models):
        model.to(device)
        model.eval()
        with torch.no_grad():
            labels, predictions, _ = get_model_predictions(data_loader, model, device)
        model.cpu()
        labels = labels.cpu()
        all_predictions[i] = predictions.cpu()
    predictions_matrix = (
        torch.stack(list(all_predictions.values()))
        * ensemble_models_weights[:, None, None]
    )

    predictions_flat = torch.Tensor(predictions_matrix).argmax(dim=2).mode(dim=0).values
    f_score = calculate_basic_metrics(
        labels_cpu=labels,
        predictions_flat=predictions_flat,
        evaluation_metric_weighting=model_specs.evaluation_metric_weighting,
        logger=logger,
    )
    return ValidationMetrics(
        f_score=f_score, loss=None, true=labels, predictions=predictions_flat
    )

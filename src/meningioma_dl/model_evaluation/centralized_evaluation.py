import logging
from pathlib import Path
from typing import Optional, Callable, Any, Tuple

import torch
from torch.utils.data import DataLoader

from meningioma_dl.config import Config
from meningioma_dl.data_loading.data_loader import get_data_loader, TransformationsMode
from meningioma_dl.experiments_specs.model_specs import ModelSpecs
from meningioma_dl.experiments_specs.preprocessing_specs import PreprocessingSpecs
from meningioma_dl.experiments_specs.training_specs import CentralizedTrainingSpecs
from meningioma_dl.model_evaluation.metrics import calculate_basic_metrics
from meningioma_dl.model_training.predictions import get_model_predictions
from meningioma_dl.models.resnet import load_model_from_file, ResNet
from meningioma_dl.visualizations.images_visualization import (
    create_images_errors_report,
)
from meningioma_dl.visualizations.results_visualizations import (
    ValidationMetrics,
    create_evaluation_report,
)


def centralized_evaluation(
    trained_model_path: Path,
    test_data_path: Path,
    config: Config,
    run_id: str,
    device: torch.device = torch.device("cpu"),
    visualizations_folder: Path = Path("."),
    model_specs: ModelSpecs = ModelSpecs(),
    preprocessing_specs: PreprocessingSpecs = PreprocessingSpecs(),
    training_specs: CentralizedTrainingSpecs = CentralizedTrainingSpecs(),
    logger: Callable[[str], None] = logging.info,
) -> float:
    logger(
        "Starting model evaluation. "
        f"Samples to be used are read from {test_data_path}"
    )
    data_loader, labels = get_data_loader(
        labels_file_path=test_data_path,
        data_root_directory=config.data_directory,
        transformations_mode=TransformationsMode.ONLY_PREPROCESSING,
        batch_size=training_specs.batch_size,
        preprocessing_specs=preprocessing_specs,
        class_mapping=model_specs.class_mapping,
        client_specific_preprocessing=training_specs.client_specific_preprocessing,
    )

    model = load_model_from_file(trained_model_path, model_specs, device)

    f_score, _ = evaluate(
        data_loader=data_loader,
        model=model,
        model_specs=model_specs,
        training_specs=training_specs,
        device=device,
        run_id=run_id,
        visualizations_folder=visualizations_folder,
        logger=logger,
    )
    return f_score


def evaluate(
    data_loader: DataLoader,
    model: ResNet,
    model_specs: ModelSpecs,
    training_specs: Any,
    device: torch.device,
    run_id: str,
    visualizations_folder: Optional[Path] = None,
    logger: Callable[[str], None] = logging.info,
    loss_function: Optional[Callable] = None,
    save_images: bool = True,
) -> Tuple[float, ValidationMetrics]:
    loss = None

    model.eval()
    with torch.no_grad():
        labels, predictions, images_paths = get_model_predictions(
            data_loader, model, device
        )
        if loss_function is not None:
            loss = (
                loss_function(
                    predictions.to(torch.float64),
                    labels.to(torch.int64).to(device),
                )
                .cpu()
                .item()
            )
    labels = labels.cpu()
    predictions_flat = predictions.cpu().argmax(dim=1)
    f_score = calculate_basic_metrics(
        labels_cpu=labels,
        predictions_flat=predictions_flat,
        evaluation_metric_weighting=model_specs.evaluation_metric_weighting,
        logger=logger,
    )
    if visualizations_folder is not None:
        create_evaluation_report(
            true=labels,
            predictions=predictions_flat,
            visualizations_folder=visualizations_folder,
            run_id=run_id,
            model_specs=model_specs,
            training_specs=training_specs,
            n_classes=model_specs.number_of_classes,
        )
        if save_images:
            create_images_errors_report(
                data_loader=data_loader,
                images_paths=images_paths,
                predictions=predictions_flat,
                directory=visualizations_folder.joinpath(
                    "evaluation_images_with_predictions"
                ),
            )
    return f_score, ValidationMetrics(
        f_score=f_score,
        loss=loss,
        true=labels.numpy(),
        predictions=predictions_flat.numpy(),
    )

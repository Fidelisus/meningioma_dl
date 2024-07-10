from sklearn.metrics import f1_score, recall_score, precision_score
import torch

import logging
from pathlib import Path
from typing import Optional, Union, Any, Callable, Tuple

import fire
from torch.utils.data import DataLoader

from meningioma_dl.config import Config
from meningioma_dl.data_loading.data_loader import (
    get_data_loader,
    TransformationsMode,
)
from meningioma_dl.experiments_specs.model_specs import ModelSpecs
from meningioma_dl.experiments_specs.preprocessing_specs import PreprocessingSpecs
from meningioma_dl.experiments_specs.training_specs import (
    CentralizedTrainingSpecs,
    get_training_specs,
)
from meningioma_dl.models.resnet import RESNET_MODELS_MAP, ResNet
from meningioma_dl.training_utils import (
    get_model_predictions,
)
from meningioma_dl.utils import (
    select_device,
    setup_logging,
    setup_run,
)
from meningioma_dl.visualizations.images_visualization import (
    create_images_errors_report,
)
from meningioma_dl.visualizations.results_visualizations import (
    create_evaluation_report,
    ValidationMetrics,
)
from meningioma_dl.federated_learning.federated_training_utils import (
    load_best_model,
)


def evaluate(
    trained_model_path: str,
    test_data_path: Path,
    run_id: Optional[str] = None,
    manual_seed: int = 123,
    device: torch.device = torch.device("cpu"),
    visualizations_folder: Path = Path("."),
    model_specs: ModelSpecs = ModelSpecs(),
    preprocessing_specs: PreprocessingSpecs = PreprocessingSpecs(),
    training_specs: CentralizedTrainingSpecs = CentralizedTrainingSpecs(),
    logger: Callable[[str], None] = logging.info,
) -> float:
    logger("Starting model evaluation")
    torch.manual_seed(manual_seed)

    logger(f"Samples to be used are read from {test_data_path}")
    data_loader, labels = get_data_loader(
        test_data_path,
        Config.data_directory,
        transformations_mode=TransformationsMode.ONLY_PREPROCESSING,
        batch_size=training_specs.batch_size,
        preprocessing_specs=preprocessing_specs,
        class_mapping=model_specs.class_mapping,
        client_specific_preprocessing=training_specs.client_specific_preprocessing,
    )

    model = load_model_for_evaluation(trained_model_path, model_specs, device)

    f_score, _ = evaluate_model(
        data_loader,
        model,
        model_specs,
        training_specs,
        device,
        run_id,
        visualizations_folder,
        logger,
    )

    return f_score


def evaluate_model(
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
            loss = loss_function(
                predictions.to(torch.float64),
                labels.to(torch.int64).to(device),
            ).cpu()
    labels_cpu = labels.cpu()
    predictions_flat = predictions.cpu().argmax(dim=1)
    f_score = calculate_basic_metrics(
        labels_cpu, predictions_flat, model_specs.evaluation_metric_weighting, logger
    )
    if visualizations_folder is not None:
        create_evaluation_report(
            labels_cpu,
            predictions_flat,
            visualizations_folder,
            run_id,
            model_specs,
            training_specs,
            model_specs.number_of_classes,
        )
        if save_images:
            create_images_errors_report(
                data_loader,
                images_paths,
                predictions_flat,
                visualizations_folder.joinpath("evaluation_images_with_predictions"),
            )
    return f_score, ValidationMetrics(f_score, loss, labels_cpu, predictions_flat)


def load_model_for_evaluation(
    trained_model_path: str, model_specs: ModelSpecs, device: torch.device
) -> ResNet:
    saved_model = torch.load(trained_model_path, map_location=device)
    no_cuda = False if device == torch.device("cuda") else True
    model = RESNET_MODELS_MAP[model_specs.model_depth](
        shortcut_type=model_specs.resnet_shortcut_type,
        no_cuda=no_cuda,
        num_classes=model_specs.number_of_classes,
    ).to(device)
    # TODO don't use hasattr
    if hasattr(saved_model["state_dict"], "items"):
        state_dict = {
            k.replace("module.", ""): v for k, v in saved_model["state_dict"].items()
        }
        model.load_state_dict(state_dict)
    else:
        model = load_best_model(model, trained_model_path, device)
    return model


def calculate_basic_metrics(
    labels_cpu: torch.Tensor,
    predictions_flat: torch.Tensor,
    evaluation_metric_weighting: str,
    logger: Callable[[str], None] = logging.info,
) -> float:
    f_score = f1_score(
        labels_cpu,
        predictions_flat,
        average=evaluation_metric_weighting,
    )
    recall = recall_score(
        labels_cpu,
        predictions_flat,
        average=evaluation_metric_weighting,
    )
    precision = precision_score(
        labels_cpu,
        predictions_flat,
        average=evaluation_metric_weighting,
    )
    logger(f"Evaluation f-score: {f_score}")
    logger(f"Evaluation recall: {recall}")
    logger(f"Evaluation precision: {precision}")
    return f_score


def run_standalone_evaluate(
    trained_model_path: str,
    env_file_path: Optional[str] = None,
    run_id: Optional[str] = None,
    device_name: str = "cpu",
    preprocessing_specs_name: str = "no_resize",
    model_specs_name: str = "resnet_10_0_unfreezed",
    use_test_data: bool = False,
    cv_fold: Optional[int] = None,
    manual_seed: int = 123,
):
    device, run_id = setup_run(env_file_path, run_id, manual_seed, device_name, cv_fold)

    labels_file = (
        Config.test_labels_file_path
        if use_test_data
        else Config.validation_labels_file_path
    )
    visualizations_folder = Config.visualizations_directory.joinpath(run_id)
    evaluate(
        trained_model_path=trained_model_path,
        device=device,
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

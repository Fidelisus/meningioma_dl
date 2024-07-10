from sklearn.metrics import f1_score, recall_score, precision_score
import torch

import logging
from pathlib import Path
from typing import Optional, Union, Any, Callable, Tuple, List

import fire
from torch.utils.data import DataLoader

from meningioma_dl.config import Config
from meningioma_dl.data_loading.data_loader import (
    get_data_loader,
    TransformationsMode,
)
from meningioma_dl.evaluate import calculate_basic_metrics, load_model_for_evaluation
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


AVAILABLE_ENSEMBLES = {
    "centralized_2_classes": (
        "cv_final_model_centralized_central_300_epochs_no_resize_with_bias_correction_1p_001_lr_099_gamma_class_2_and_3_together_4_unfreezed_fold0_6376843",
        "cv_final_model_centralized_central_300_epochs_no_resize_with_bias_correction_1p_001_lr_099_gamma_class_2_and_3_together_4_unfreezed_fold1_6376844",
        "cv_final_model_centralized_central_300_epochs_no_resize_with_bias_correction_1p_001_lr_099_gamma_class_2_and_3_together_4_unfreezed_fold2_6376845",
        "cv_final_model_centralized_central_300_epochs_no_resize_with_bias_correction_1p_001_lr_099_gamma_class_2_and_3_together_4_unfreezed_fold3_6376846",
        "cv_final_model_centralized_central_300_epochs_no_resize_with_bias_correction_1p_001_lr_099_gamma_class_2_and_3_together_4_unfreezed_fold4_6376847",
    ),
    "centralized_3_classes": (
        "cv_final_model_centralized_central_300_epochs_no_resize_with_bias_correction_08p_0001_lr_099_gamma_resnet_10_4_unfreezed_fold0_6376848",
        "cv_final_model_centralized_central_300_epochs_no_resize_with_bias_correction_08p_0001_lr_099_gamma_resnet_10_4_unfreezed_fold1_6376849",
        "cv_final_model_centralized_central_300_epochs_no_resize_with_bias_correction_08p_0001_lr_099_gamma_resnet_10_4_unfreezed_fold2_6384955",
        "cv_final_model_centralized_central_300_epochs_no_resize_with_bias_correction_08p_0001_lr_099_gamma_resnet_10_4_unfreezed_fold3_6384956",
        "cv_final_model_centralized_central_300_epochs_no_resize_with_bias_correction_08p_0001_lr_099_gamma_resnet_10_4_unfreezed_fold4_6384957",
    ),
    "local": (
        "cv_final_model_centralized_central_300_epochs_no_resize_with_bias_correction_1p_001_lr_099_gamma_class_2_and_3_together_4_unfreezed_fold0_6376843",
        "cv_final_model_centralized_central_300_epochs_no_resize_with_bias_correction_1p_001_lr_099_gamma_class_2_and_3_together_4_unfreezed_fold1_6376844",
        "cv_final_model_centralized_central_300_epochs_no_resize_with_bias_correction_1p_001_lr_099_gamma_class_2_and_3_together_4_unfreezed_fold2_6376845",
    ),
}


def evaluate_ensemble(
    trained_model_paths: List[str],
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
    logger(
        f"Starting enseble model evaluation using the models from {trained_model_paths}"
    )
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

    models: List[ResNet] = []
    for trained_model_path in trained_model_paths:
        model = load_model_for_evaluation(
            trained_model_path, model_specs, torch.device("cpu")
        )
        models.append(model)

    f_score, _ = evaluate_model(
        data_loader,
        models,
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
    models: List[ResNet],
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

    if not models:
        raise ValueError("More than one model is needed to evaluate an ensemble.")

    all_predictions = []
    for i, model in enumerate(models):
        model.to(device)
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
        model.cpu()
        labels_cpu = labels.cpu()
        all_predictions.append(predictions.cpu().argmax(dim=1))

    predictions_flat = torch.stack(all_predictions).mode(dim=0).values
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


def run_ensemble_evaluate(
    ensemble_id: str,
    env_file_path: Optional[str] = None,
    run_id: Optional[str] = None,
    device_name: str = "cpu",
    preprocessing_specs_name: str = "no_resize",
    model_specs_name: str = "class_2_and_3_together_4_unfreezed",
    manual_seed: int = 123,
):
    device, run_id = setup_run(env_file_path, run_id, manual_seed, device_name, None)
    trained_model_paths = []
    for model_id in AVAILABLE_ENSEMBLES[ensemble_id]:
        trained_model_paths.append(
            str(Config.saved_models_directory.joinpath(model_id, "epoch_-1.pth.tar"))
        )
    evaluate_ensemble(
        trained_model_paths=trained_model_paths,
        device=device,
        visualizations_folder=Config.visualizations_directory.joinpath(run_id),
        model_specs=ModelSpecs.get_from_name(model_specs_name),
        preprocessing_specs=PreprocessingSpecs.get_from_name(preprocessing_specs_name),
        training_specs=get_training_specs("evaluation"),
        test_data_path=Config.test_labels_file_path,
    )


if __name__ == "__main__":
    try:
        fire.Fire(run_ensemble_evaluate)
    except Exception as e:
        logging.error(f"An unexpected error occured {e}", exc_info=True)

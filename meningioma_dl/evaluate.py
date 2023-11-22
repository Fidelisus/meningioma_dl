import logging
from pathlib import Path
from typing import Optional, Union

import fire
import torch
from sklearn.metrics import f1_score, recall_score, precision_score

from meningioma_dl.config import Config
from meningioma_dl.data_loading.data_loader import (
    get_data_loader,
    TransformationsMode,
)
from meningioma_dl.experiments_specs.modelling_specs import ModellingSpecs
from meningioma_dl.experiments_specs.training_specs import CentralizedTrainingSpecs
from meningioma_dl.models.resnet import RESNET_MODELS_MAP
from meningioma_dl.training_utils import get_model_predictions
from meningioma_dl.utils import (
    select_device,
    setup_logging,
)
from meningioma_dl.visualizations.images_visualization import (
    create_images_errors_report,
)
from meningioma_dl.visualizations.results_visualizations import create_evaluation_report


def evaluate(
    trained_model_path: str,
    env_file_path: Optional[str] = None,
    run_id: Optional[str] = None,
    manual_seed: int = Config.random_seed,
    device_name: str = "cpu",
    visualizations_folder: Union[str, Path] = Path("."),
    modelling_specs: ModellingSpecs = ModellingSpecs(),
    training_specs: CentralizedTrainingSpecs = CentralizedTrainingSpecs(),
) -> float:
    if type(visualizations_folder) is str:
        visualizations_folder = Path(visualizations_folder)
    if env_file_path is not None:
        Config.load_env_variables(
            env_file_path, f"evaluation_{Path(trained_model_path).parent}"
        )
        setup_logging(Config.log_file_path)

    logging.info("Starting model evaluation")

    device = select_device(device_name)
    torch.manual_seed(manual_seed)

    data_loader, labels = get_data_loader(
        Config.train_labels_file_path
        if training_specs.use_training_data_for_validation
        else Config.validation_labels_file_path,
        Config.data_directory,
        transformations_mode=TransformationsMode.ONLY_PREPROCESSING,
        batch_size=training_specs.batch_size,
        preprocessing_specs=modelling_specs.preprocessing_specs,
        class_mapping=modelling_specs.model_specs.class_mapping,
    )

    saved_model = torch.load(trained_model_path, map_location=device)
    no_cuda = False if device == torch.device("cuda") else True
    model = RESNET_MODELS_MAP[modelling_specs.model_specs.model_depth](
        shortcut_type=modelling_specs.model_specs.resnet_shortcut_type,
        no_cuda=no_cuda,
        num_classes=modelling_specs.model_specs.number_of_classes,
    ).to(device)
    state_dict = {
        k.replace("module.", ""): v for k, v in saved_model["state_dict"].items()
    }
    model.load_state_dict(state_dict)

    model.eval()
    with torch.no_grad():
        labels, predictions, images_paths = get_model_predictions(
            data_loader, model, device
        )

    labels_cpu = labels.cpu()
    predictions_flat = predictions.cpu().argmax(dim=1)

    f_score = f1_score(
        labels_cpu,
        predictions_flat,
        average=modelling_specs.model_specs.evaluation_metric_weighting,
    )
    recall = recall_score(
        labels_cpu,
        predictions_flat,
        average=modelling_specs.model_specs.evaluation_metric_weighting,
    )
    precision = precision_score(
        labels_cpu,
        predictions_flat,
        average=modelling_specs.model_specs.evaluation_metric_weighting,
    )

    logging.info(f"Evaluation f-score: {f_score}")
    logging.info(f"Evaluation recall: {recall}")
    logging.info(f"Evaluation precision: {precision}")

    create_evaluation_report(
        labels_cpu,
        predictions_flat,
        visualizations_folder,
        run_id,
        modelling_specs,
        training_specs,
    )
    create_images_errors_report(
        data_loader,
        images_paths,
        predictions_flat,
        visualizations_folder.joinpath("evaluation_images_with_predictions"),
    )

    return f_score


if __name__ == "__main__":
    try:
        fire.Fire(evaluate)
    except Exception as e:
        logging.error(f"An unexpected error occured {e}", exc_info=True)

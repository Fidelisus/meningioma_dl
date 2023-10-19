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
    PreprocessingSettings,
)
from meningioma_dl.models.resnet import RESNET_MODELS_MAP
from meningioma_dl.training_utils import get_model_predictions
from meningioma_dl.utils import (
    select_device,
    setup_logging,
)
from meningioma_dl.visualizations.results_visualizations import plot_confusion_matrix


def evaluate(
    trained_model_path: str,
    env_file_path: Optional[str] = None,
    manual_seed: int = Config.random_seed,
    model_depth: int = 10,
    resnet_shortcut_type: str = "B",
    num_workers: int = 1,
    number_of_classes: int = 3,
    device_name: str = "cpu",
    visualizations_folder: Union[str, Path] = Path("."),
    batch_size: int = 1,
    preprocessing_settings: PreprocessingSettings = PreprocessingSettings(),
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
        Config.validation_labels_file_path,
        Config.data_directory,
        num_workers,
        transformations_mode=TransformationsMode.ONLY_PREPROCESSING,
        batch_size=batch_size,
        preprocessing_settings=preprocessing_settings,
    )

    saved_model = torch.load(trained_model_path, map_location=device)
    no_cuda = False if device == torch.device("cuda") else True
    model = RESNET_MODELS_MAP[model_depth](
        shortcut_type=resnet_shortcut_type,
        no_cuda=no_cuda,
        num_classes=number_of_classes,
    ).to(device)
    state_dict = {
        k.replace("module.", ""): v for k, v in saved_model["state_dict"].items()
    }
    model.load_state_dict(state_dict)

    model.eval()
    with torch.no_grad():
        labels, predictions = get_model_predictions(data_loader, model, device)

    labels_cpu = labels.cpu()
    predictions_flat = predictions.cpu().argmax(dim=1)

    f_score = f1_score(
        labels_cpu,
        predictions_flat,
        average="weighted",
    )
    recall = recall_score(
        labels_cpu,
        predictions_flat,
        average="weighted",
    )
    precision = precision_score(
        labels_cpu,
        predictions_flat,
        average="weighted",
    )

    logging.info(f"Evaluation f-score: {f_score}")
    logging.info(f"Evaluation recall: {recall}")
    logging.info(f"Evaluation precision: {precision}")

    plot_confusion_matrix(labels_cpu, predictions_flat, visualizations_folder)

    return f_score


if __name__ == "__main__":
    try:
        fire.Fire(evaluate)
    except Exception as e:
        logging.error(f"An unexpected error occured {e}", exc_info=True)

import logging
from pathlib import Path
from typing import Tuple

import fire
import torch
from sklearn.metrics import f1_score, recall_score, precision_score

from meningioma_dl.config import Config
from meningioma_dl.data_loading.data_loader import get_data_loader, TransformationsMode
from meningioma_dl.models.resnet import RESNET_MODELS_MAP
from meningioma_dl.training_utils import get_model_predictions
from meningioma_dl.utils import select_device, get_loss_function_class_weights
from meningioma_dl.visualizations.results_visualizations import plot_confusion_matrix


def evaluate(
    manual_seed: int = Config.random_seed,
    model_depth: int = 10,
    resnet_shortcut_type: str = "B",
    num_workers: int = 1,
    number_of_classes: int = 3,
    gpus_ids: Tuple[int] = (),
    trained_model_path: Path = Path("trails/models/current_model_epoch_0.pth.tar"),
    device_name: str = "cpu",
    ci_run: bool = True,
):
    logging.info("Starting model evaluation")

    device = select_device(device_name)

    torch.manual_seed(manual_seed)

    if ci_run:
        labels_file_path = Config.ci_run_labels_file_path
        data_root_directory = Config.ci_images_directory
    else:
        labels_file_path = Config.validation_labels_file_path
        data_root_directory = Config.images_directory
    data_loader, labels = get_data_loader(
        labels_file_path,
        data_root_directory,
        num_workers,
        transformations_mode=TransformationsMode.ONLY_PREPROCESSING,
    )

    loss_function_class_weights = get_loss_function_class_weights(labels)

    saved_model = torch.load(trained_model_path)
    no_cuda = False if device == torch.device("cuda") else True
    model = RESNET_MODELS_MAP[model_depth](
        shortcut_type=resnet_shortcut_type,
        no_cuda=no_cuda,
        num_classes=number_of_classes,
    ).to(device)
    model.load_state_dict(saved_model["state_dict"])

    labels, predictions = get_model_predictions(data_loader, model, device)

    labels_flat = labels.argmax(dim=1)
    predictions_flat = predictions.argmax(dim=1)

    f_score = f1_score(
        labels_flat,
        predictions_flat,
        sample_weight=loss_function_class_weights,
        average="weighted",
    )
    recall = recall_score(
        labels_flat,
        predictions_flat,
        sample_weight=loss_function_class_weights,
        average="weighted",
    )
    precision = precision_score(
        labels_flat,
        predictions_flat,
        sample_weight=loss_function_class_weights,
        average="weighted",
    )

    logging.info(f"Evaluation f-score: {f_score}")
    logging.info(f"Evaluation recall: {recall}")
    logging.info(f"Evaluation precision: {precision}")

    plot_confusion_matrix(
        labels_flat, predictions_flat, Config.visualizations_directory
    )


if __name__ == "__main__":
    try:
        fire.Fire(evaluate)
    except Exception as e:
        logging.error(f"An unexpected error occured {e}", exc_info=True)

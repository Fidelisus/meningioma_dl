import logging
from pathlib import Path
from typing import Tuple, Optional

import fire
import torch
from torch import nn

from meningioma_dl.config import Config
from meningioma_dl.data_loading.data_loader import (
    get_data_loader,
    TransformationsMode,
)
from meningioma_dl.experiments_specs.modelling_specs import ModellingSpecs
from meningioma_dl.experiments_specs.training_specs import CentralizedTrainingSpecs
from meningioma_dl.models.resnet import create_resnet_model, freeze_layers
from meningioma_dl.training_utils import training_loop
from meningioma_dl.utils import (
    get_loss_function_class_weights,
)


def train(
    manual_seed: int = 123,
    validation_interval: int = 1,
    device: torch.device = torch.device("cpu"),
    saved_models_folder: Path = Path("."),
    visualizations_folder: Path = Path("."),
    modelling_specs: ModellingSpecs = ModellingSpecs(),
    training_specs: CentralizedTrainingSpecs = CentralizedTrainingSpecs(),
) -> Tuple[float, Optional[str]]:
    logging.info(
        f"Starting training with {modelling_specs.model_specs.number_of_classes} classes"
    )
    torch.manual_seed(manual_seed)
    train_labels_file_path = Config.train_labels_file_path
    validation_labels_file_path = Config.validation_labels_file_path
    training_data_loader, labels_train = get_data_loader(
        train_labels_file_path,
        Config.data_directory,
        transformations_mode=TransformationsMode.AUGMENT,
        batch_size=training_specs.batch_size,
        augmentations=modelling_specs.augmentation_specs.transformations_list,
        preprocessing_specs=modelling_specs.preprocessing_specs,
        class_mapping=modelling_specs.model_specs.class_mapping,
    )
    logging.info(f"Training data loaded from {train_labels_file_path}")
    validation_data_loader, labels_validation = get_data_loader(
        validation_labels_file_path,
        Config.data_directory,
        transformations_mode=TransformationsMode.ONLY_PREPROCESSING,
        batch_size=training_specs.batch_size,
        preprocessing_specs=modelling_specs.preprocessing_specs,
        class_mapping=modelling_specs.model_specs.class_mapping,
    )
    logging.info(f"Validation data loaded from {validation_labels_file_path}")

    model, pretrained_model_state_dict = create_resnet_model(
        modelling_specs.model_specs.model_depth,
        modelling_specs.model_specs.resnet_shortcut_type,
        modelling_specs.model_specs.number_of_classes,
        Config.pretrained_models_directory,
        device,
    )
    parameters_names_to_fine_tune, parameters_to_fine_tune = freeze_layers(
        model,
        modelling_specs.model_specs.number_of_layers_to_unfreeze,
        pretrained_model_state_dict,
    )
    logging.info(f"Parameters to fine tune: {parameters_names_to_fine_tune}")

    lr_params = [
        {
            "params": parameters_to_fine_tune,
            "lr": modelling_specs.scheduler_specs.learning_rate,
        }
    ]
    optimizer = torch.optim.Adam(lr_params)
    scheduler = modelling_specs.scheduler_specs.get_scheduler(optimizer)

    loss_function_weighting = (
        torch.tensor(
            get_loss_function_class_weights(labels_train + labels_validation)
        ).to(torch.float64)
        if modelling_specs.model_specs.evaluation_metric_weighting == "weighted"
        else None
    )
    loss_function = nn.CrossEntropyLoss(
        weight=loss_function_weighting,
    )
    loss_function = loss_function.to(device)

    logging.info("Model initialized succesfully")

    best_f_score, trained_model_path, _ = training_loop(
        training_data_loader,
        validation_data_loader,
        model,
        optimizer,
        scheduler,
        loss_function,
        total_epochs=training_specs.epochs,
        validation_interval=validation_interval,
        model_save_folder=saved_models_folder,
        visualizations_folder=visualizations_folder,
        device=device,
        evaluation_metric_weighting=modelling_specs.model_specs.evaluation_metric_weighting,
    )
    return best_f_score, trained_model_path


if __name__ == "__main__":
    try:
        fire.Fire(train)
    except Exception as e:
        logging.error(f"An unexpected error occured {e}", exc_info=True)
        raise

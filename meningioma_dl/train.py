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
from meningioma_dl.experiments_specs.experiments import ModellingSpecs
from meningioma_dl.experiments_specs.traning_specs import CentralizedTrainingSpecs
from meningioma_dl.models.resnet import create_resnet_model
from meningioma_dl.training_utils import training_loop
from meningioma_dl.utils import (
    select_device,
    get_loss_function_class_weights,
    setup_logging,
    generate_run_id,
)


def train(
    env_file_path: Optional[str] = None,
    run_id: Optional[str] = None,
    manual_seed: int = Config.random_seed,
    validation_interval: int = 1,
    device_name: str = "cpu",
    saved_models_folder: Path = Path("."),
    visualizations_folder: Path = Path("."),
    modelling_specs: ModellingSpecs = ModellingSpecs(),
    training_specs: CentralizedTrainingSpecs = CentralizedTrainingSpecs(),
) -> Tuple[float, Optional[str]]:
    if run_id is None:
        run_id = generate_run_id()

    if env_file_path is not None:
        Config.load_env_variables(env_file_path, run_id)
        setup_logging(Config.log_file_path)

    device = get_device(device_name)
    torch.manual_seed(manual_seed)

    logging.info("Start training")
    training_data_loader, labels_train = get_data_loader(
        Config.train_labels_file_path,
        Config.data_directory,
        transformations_mode=TransformationsMode.AUGMENT,
        batch_size=training_specs.batch_size,
        augmentations=modelling_specs.augmentation_specs.transformations_list,
        preprocessing_specs=modelling_specs.preprocessing_specs,
    )
    validation_data_loader, labels_validation = get_data_loader(
        Config.train_labels_file_path
        if training_specs.use_training_data_for_validation
        else Config.validation_labels_file_path,
        Config.data_directory,
        transformations_mode=TransformationsMode.ONLY_PREPROCESSING,
        batch_size=training_specs.batch_size,
        preprocessing_specs=modelling_specs.preprocessing_specs,
    )

    model, parameters_to_fine_tune = create_resnet_model(
        modelling_specs.model_specs.model_depth,
        modelling_specs.model_specs.resnet_shortcut_type,
        modelling_specs.model_specs.number_of_classes,
        Config.pretrained_models_directory,
        device,
        modelling_specs.model_specs.number_of_layers_to_unfreeze,
    )

    lr_params = [
        {
            "params": parameters_to_fine_tune,
            "lr": modelling_specs.scheduler_specs.learning_rate,
        }
    ]
    optimizer = torch.optim.Adam(lr_params)
    scheduler = modelling_specs.scheduler_specs.get_scheduler(optimizer)

    loss_function = nn.CrossEntropyLoss(
        weight=torch.tensor(
            get_loss_function_class_weights(labels_train + labels_validation)
        ).to(torch.float64),
    )
    loss_function = loss_function.to(device)

    logging.info("Model initialized succesfully")

    best_f_score, trained_model_path = training_loop(
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
    )
    return best_f_score, trained_model_path


def get_device(device_name) -> torch.device:
    device = select_device(device_name)
    logging.info("Devices available:")
    torch.cuda.is_available()
    for device_number in range(torch.cuda.device_count()):
        logging.info(torch.cuda.get_device_name(device_number))
    return device


if __name__ == "__main__":
    try:
        fire.Fire(train)
    except Exception as e:
        logging.error(f"An unexpected error occured {e}", exc_info=True)
        raise

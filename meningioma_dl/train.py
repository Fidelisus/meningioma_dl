import logging
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any

import fire
import torch
from monai import transforms
from torch import optim, nn

from meningioma_dl.config import Config
from meningioma_dl.data_loading.data_loader import (
    get_data_loader,
    TransformationsMode,
    PreprocessingSettings,
)
from meningioma_dl.models.resnet import create_resnet_model
from meningioma_dl.training_utils import training_loop, WeightedKappaLoss
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
    augmentation_settings: Optional[List[transforms.Transform]] = None,
    learning_rate: float = 0.1,
    weight_decay: float = 0.0,
    scheduler_parameters: Dict[str, Any] = {},
    model_depth: int = 10,
    batch_size: int = 2,
    n_epochs: int = 2,
    resnet_shortcut_type: str = "B",
    validation_interval: int = 1,
    n_workers: int = 1,
    number_of_classes: int = 3,
    device_name: str = "cpu",
    save_intermediate_models: bool = False,
    saved_models_folder: Path = Path("."),
    visualizations_folder: Path = Path("."),
    scheduler=optim.lr_scheduler.ExponentialLR,
    preprocessing_settings: PreprocessingSettings = PreprocessingSettings(),
    freeze_all_layers: bool = True,
    use_training_data_for_validation: bool = False,
    loss_function_name: str = "cross_entropy",
) -> Tuple[float, Optional[str]]:
    if run_id is None:
        run_id = generate_run_id()

    if env_file_path is not None:
        Config.load_env_variables(env_file_path, run_id)
        setup_logging(Config.log_file_path)

    device = select_device(device_name)

    logging.info("Devices available:")
    torch.cuda.is_available()
    for device_number in range(torch.cuda.device_count()):
        logging.info(torch.cuda.get_device_name(device_number))

    torch.manual_seed(manual_seed)

    logging.info("Start training")

    training_data_loader, labels_train = get_data_loader(
        Config.train_labels_file_path,
        Config.data_directory,
        n_workers,
        transformations_mode=TransformationsMode.AUGMENT,
        batch_size=batch_size,
        augmentation_settings=augmentation_settings,
        preprocessing_settings=preprocessing_settings,
    )
    validation_data_loader, labels_validation = get_data_loader(
        Config.train_labels_file_path
        if use_training_data_for_validation
        else Config.validation_labels_file_path,
        Config.data_directory,
        n_workers,
        transformations_mode=TransformationsMode.ONLY_PREPROCESSING,
        batch_size=batch_size,
        preprocessing_settings=preprocessing_settings,
    )

    loss_function_class_weights = get_loss_function_class_weights(
        labels_train + labels_validation
    )

    (
        model,
        pretrained_model_parameters,
        classifier_model_parameters,
    ) = create_resnet_model(
        model_depth,
        resnet_shortcut_type,
        number_of_classes,
        Config.pretrained_models_directory,
        device,
        freeze_all_layers,
    )

    lr_params = [
        {"params": classifier_model_parameters, "lr": learning_rate},
    ]
    if freeze_all_layers:
        lr_params.append(
            {"params": pretrained_model_parameters, "lr": learning_rate / 5.0}
        )
    optimizer = torch.optim.Adam(lr_params, weight_decay=weight_decay)
    scheduler = scheduler(optimizer, **scheduler_parameters)
    if loss_function_name == "cross_entropy":
        loss_function = nn.CrossEntropyLoss(
            weight=torch.tensor(loss_function_class_weights).to(torch.float64),
        )
    else:
        loss_function = WeightedKappaLoss(3)
    loss_function = loss_function.to(device)

    logging.info("Model initialized succesfully")

    best_f_score, trained_model_path = training_loop(
        training_data_loader,
        validation_data_loader,
        model,
        optimizer,
        scheduler,
        loss_function,
        total_epochs=n_epochs,
        validation_interval=validation_interval,
        save_intermediate_models=save_intermediate_models,
        model_save_folder=saved_models_folder,
        visualizations_folder=visualizations_folder,
        device=device,
    )
    return best_f_score, trained_model_path


if __name__ == "__main__":
    try:
        fire.Fire(train)
    except Exception as e:
        logging.error(f"An unexpected error occured {e}", exc_info=True)
        raise

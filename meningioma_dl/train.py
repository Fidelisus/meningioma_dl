import logging
from typing import Tuple, Optional

import fire
import torch
from monai import transforms
from torch import optim

from meningioma_dl.config import Config
from meningioma_dl.data_loading.data_loader import get_data_loader, TransformationsMode
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
    augmentation_settings: Optional[list[transforms.Transform]] = None,
    learning_rate: float = 0.001,
    sgd_momentum: float = 0.9,
    weight_decay: float = 0.001,
    lr_scheduler_gamma: float = 0.99,
    model_depth: int = 10,
    batch_size: int = 2,
    n_epochs: int = 2,
    resnet_shortcut_type: str = "B",
    validation_interval: int = 1,
    n_workers: int = 1,
    number_of_classes: int = 3,
    device_name: str = "cpu",
) -> tuple[float, Optional[str]]:
    if run_id is None:
        run_id = generate_run_id()

    if env_file_path is not None:
        Config.load_env_variables(env_file_path, run_id)
        setup_logging(Config.log_file_path)

    device = select_device(device_name)
    torch.manual_seed(manual_seed)

    logging.info("Start training")

    training_data_loader, labels_train = get_data_loader(
        Config.train_labels_file_path,
        Config.data_directory,
        n_workers,
        transformations_mode=TransformationsMode.AUGMENT,
        batch_size=batch_size,
        augmentation_settings=augmentation_settings,
    )
    validation_data_loader, labels_validation = get_data_loader(
        Config.validation_labels_file_path,
        Config.data_directory,
        n_workers,
        transformations_mode=TransformationsMode.ONLY_PREPROCESSING,
        batch_size=batch_size,
    )

    loss_function_class_weights = get_loss_function_class_weights(
        labels_train + labels_validation
    )

    model, pretrained_model_parameters, parameters_to_fine_tune = create_resnet_model(
        model_depth,
        resnet_shortcut_type,
        number_of_classes,
        Config.pretrained_models_directory,
        device,
    )

    params = [
        {"params": pretrained_model_parameters, "lr": learning_rate},
        {"params": parameters_to_fine_tune, "lr": learning_rate * 100},
    ]
    optimizer = torch.optim.SGD(
        params, momentum=sgd_momentum, weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_scheduler_gamma)

    logging.info("Model initialized succesfully")

    # TODO train from resume
    # if sets.resume_path:
    #     if os.path.isfile(sets.resume_path):
    #         print("=> loading checkpoint '{}'".format(sets.resume_path))
    #         checkpoint = torch.load(sets.resume_path)
    #         model.load_state_dict(checkpoint["state_dict"])
    #         optimizer.load_state_dict(checkpoint["optimizer"])
    #         print(
    #             "=> loaded checkpoint '{}' (epoch {})".format(
    #                 sets.resume_path, checkpoint["epoch"]
    #             )
    #         )

    best_f_score, trained_model_path = training_loop(
        training_data_loader,
        validation_data_loader,
        model,
        optimizer,
        scheduler,
        loss_function_class_weights,
        total_epochs=n_epochs,
        validation_interval=validation_interval,
        model_save_folder=Config.saved_models_directory,
        device=device,
        run_id=run_id,
    )
    return best_f_score, trained_model_path


if __name__ == "__main__":
    try:
        fire.Fire(train)
    except Exception as e:
        logging.error(f"An unexpected error occured {e}", exc_info=True)
        raise

import logging
from pathlib import Path
from typing import Tuple, Optional

import fire
import torch
from torch import optim

from meningioma_dl.config import Config
from meningioma_dl.data_loading.data_loader import get_data_loader, TransformationsMode
from meningioma_dl.training_utils import training_loop
from meningioma_dl.utils import (
    select_device,
    get_loss_function_class_weights,
    setup_logging,
)
from model import create_resnet_model


def train(
    manual_seed: int = Config.random_seed,
    learning_rate: float = 0.001,
    sgd_momentum: float = 0.9,
    weight_decay: float = 0.001,
    lr_scheduler_gamma: float = 0.99,
    model_depth: int = 10,
    batch_size: int = 2,
    n_epochs: int = 2,
    resnet_shortcut_type: str = "B",
    validation_interval: int = 1,
    num_workers: int = 1,
    number_of_classes: int = 3,
    gpus_ids: Tuple[int] = (),
    pretrained_model_path: Optional[Path] = Path("pretrain/resnet_10.pth"),
    save_folder: Path = Path("trails/models/current_model"),  # TODO make it nicer
    device_name: str = "cpu",
    ci_run: bool = True,
):
    setup_logging()

    device = select_device(device_name)
    torch.manual_seed(manual_seed)

    if ci_run:
        labels_file_path_train = Config.ci_run_labels_file_path
        labels_file_path_validation = Config.ci_run_labels_file_path
        data_root_directory = Config.ci_run_data_root_directory
    else:
        labels_file_path_train = Config.train_labels_file_path
        labels_file_path_validation = Config.validation_labels_file_path
        data_root_directory = Config.data_root_directory

    logging.info("Start training")

    training_data_loader, labels_train = get_data_loader(
        labels_file_path_train,
        data_root_directory,
        num_workers,
        transformations_mode=TransformationsMode.AUGMENT,
        batch_size=batch_size,
    )
    validation_data_loader, labels_validation = get_data_loader(
        labels_file_path_validation,
        data_root_directory,
        num_workers,
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
        gpus_ids,
        pretrained_model_path,
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
    # sets = {}
    # sets["input_D"] = 56
    # sets["input_H"] = 448
    # sets["input_W"] = 448
    # sets["phase"] = "train"
    # sets = type("Sets", (object,), sets)()
    # training_dataset = BrainS18Dataset("data", "./data/train.txt", sets)
    #
    # data_loader = DataLoader(
    #     training_dataset,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     num_workers=num_workers,
    #     pin_memory=pin_memory,
    # )

    # training

    training_loop(
        training_data_loader,
        validation_data_loader,
        model,
        optimizer,
        scheduler,
        loss_function_class_weights,
        total_epochs=n_epochs,
        validation_interval=validation_interval,
        model_save_folder=save_folder,
        device=device,
        ci_run=ci_run,
    )


if __name__ == "__main__":
    try:
        fire.Fire(train)
    except Exception as e:
        logging.error(f"An unexpected error occured {e}", exc_info=True)
        raise

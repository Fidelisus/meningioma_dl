import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
from monai.data import DataLoader
from sklearn.metrics import f1_score
from torch import nn
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

from meningioma_dl.utils import one_hot_encode_labels


def training_loop(
    training_data_loader: DataLoader,
    validation_data_loader: DataLoader,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    loss_function_class_weights: np.array,
    total_epochs: int,
    validation_interval: int,
    model_save_folder: Path,
    device: torch.device,
    ci_run: bool,
):
    best_loss_validation = torch.tensor(np.inf)
    best_f_score = 0.0
    batches_per_epoch = len(training_data_loader)
    logging.info(f"total_epochs: {total_epochs} batches_per_epoch: {batches_per_epoch}")
    loss_function = nn.CrossEntropyLoss(
        ignore_index=-1, weight=torch.tensor(loss_function_class_weights)
    )

    loss_function = loss_function.to(device)

    for epoch in range(total_epochs):
        logging.info("Start epoch {}".format(epoch))
        epoch_start_time = time.time()
        step = 0
        epoch_loss = 0

        model.train()

        for batch_id, batch_data in enumerate(training_data_loader):
            step += 1
            inputs, labels = batch_data["img"].to(device), batch_data["label"].to(
                device
            )
            optimizer.zero_grad()
            predictions = model(inputs)
            labels_onehot = one_hot_encode_labels(labels)
            loss = loss_function(predictions, labels_onehot)
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        logging.info(f"epoch {epoch} average loss: {epoch_loss / step:.4f}")
        logging.info(f"learning rate: {scheduler.get_last_lr()}")
        logging.info(f"batch time: {time.time() - epoch_start_time}")

        if (epoch + 1) % validation_interval == 0:
            model.eval()
            with torch.no_grad():
                labels, predictions = get_model_predictions(
                    validation_data_loader, model, device
                )

                f_score = f1_score(
                    labels.argmax(dim=1),
                    predictions.argmax(dim=1),
                    sample_weight=loss_function_class_weights,
                    average="micro",
                )
                if f_score > best_f_score:
                    best_f_score = f_score
                loss_validation: torch.Tensor = loss_function(predictions, labels)

                if not ci_run and loss_validation > best_loss_validation:
                    _save_model(model, model_save_folder, optimizer, epoch)
                logging.info(f"F1 score: {f_score}")
                logging.info(f"Validation loss: {loss_validation.data}")

    logging.info(
        f"Finished training, best f_score: {best_f_score}, best validation loss: {best_loss_validation.data}"
    )


def get_model_predictions(
    validation_data_loader: DataLoader, model: nn.Module, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    predictions = torch.tensor([], dtype=torch.float32, device=device)
    labels = torch.tensor([], dtype=torch.long, device=device)
    for validation_data in validation_data_loader:
        validation_images, validation_labels = validation_data["img"].to(
            device
        ), validation_data["label"].to(device)
        predictions = torch.cat([predictions, model(validation_images)], dim=0)
        labels = torch.cat([labels, one_hot_encode_labels(validation_labels)], dim=0)
    return labels, predictions


def _save_model(
    model: nn.Module,
    model_save_folder: Path,
    optimizer: Optimizer,
    epoch: int,
):
    # TODO rewrite nicer
    # if batch_id_sp != 0 and batch_id_sp % save_interval == 0:
    model_save_path = "{}_epoch_{}.pth.tar".format(model_save_folder, epoch)
    model_save_dir = os.path.dirname(model_save_path)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    logging.info("Save checkpoints: epoch = {}".format(epoch))
    torch.save(
        {
            "ecpoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        model_save_path,
    )

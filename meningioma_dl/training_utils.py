import logging
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from monai.data import DataLoader
from sklearn.metrics import f1_score
from torch import nn
from torch.optim.optimizer import Optimizer

from meningioma_dl.utils import one_hot_encode_labels


def training_loop(
    training_data_loader: DataLoader,
    validation_data_loader: DataLoader,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler,
    loss_function_class_weights: np.array,
    total_epochs: int,
    validation_interval: int,
    model_save_folder: Path,
    device: torch.device,
    run_id: str,
) -> Tuple[float, Optional[Path]]:
    best_loss_validation = torch.tensor(np.inf)
    best_f_score = 0.0
    batches_per_epoch = len(training_data_loader)
    logging.info(f"total_epochs: {total_epochs} batches_per_epoch: {batches_per_epoch}")
    loss_function = nn.CrossEntropyLoss(
        ignore_index=-1, weight=torch.tensor(loss_function_class_weights)
    )

    loss_function = loss_function.to(device)

    trained_model_path: Optional[Path] = None
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

                if loss_validation < best_loss_validation:
                    trained_model_path = _save_model(
                        model, model_save_folder, optimizer, epoch, run_id
                    )
                    logging.info(f"Model saved at {trained_model_path}")
                logging.info(f"F1 score: {f_score}")
                logging.info(f"Validation loss: {loss_validation.data}")

    logging.info(
        f"Finished training, best f_score: {best_f_score}, best validation loss: {best_loss_validation.data}"
    )
    return best_f_score, trained_model_path


def get_model_predictions(
    validation_data_loader: DataLoader, model: nn.Module, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
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
    run_id: str,
) -> Path:
    model_save_path = model_save_folder.joinpath(run_id, f"epoch_{epoch}.pth.tar")
    model_save_path.parent.mkdir()
    logging.info(f"Saving model with run id {run_id} at epoch: {epoch}")
    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        str(model_save_path),
    )
    return model_save_path

import logging
import time
from pathlib import Path
from typing import Optional, Tuple, Union, List

import numpy as np
import torch
from monai.data import DataLoader
from sklearn.metrics import f1_score
from torch import nn
from torch.optim.optimizer import Optimizer

from meningioma_dl.config import Config
from meningioma_dl.visualizations.results_visualizations import plot_training_curve


def training_loop(
    training_data_loader: DataLoader,
    validation_data_loader: DataLoader,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler,
    loss_function_class_weights: np.array,
    total_epochs: int,
    validation_interval: int,
    visualizations_folder: Path,
    device: torch.device,
    model_save_folder: Optional[Path] = None,
) -> Tuple[float, Optional[Path]]:
    best_loss_validation = torch.tensor(np.inf)
    best_f_score = 0.0
    batches_per_epoch = len(training_data_loader)
    logging.info(f"total_epochs: {total_epochs} batches_per_epoch: {batches_per_epoch}")
    loss_function = nn.CrossEntropyLoss(
        ignore_index=-1,
        weight=torch.tensor(loss_function_class_weights).to(torch.float64),
    )

    loss_function = loss_function.to(device)

    trained_model_path: Optional[Path] = None
    training_losses: List[float] = []
    validation_losses: List[Union[float, None]] = []
    for epoch in range(total_epochs):
        logging.info("Start epoch {}".format(epoch))
        epoch_start_time = time.time()
        step = 0
        epoch_loss = 0

        model.train(True)

        for batch_id, batch_data in enumerate(training_data_loader):
            step += 1
            inputs, labels = batch_data["img"].to(device), batch_data["label"].to(
                device
            )
            optimizer.zero_grad()
            predictions = model(inputs).to(torch.float64)
            loss = loss_function(
                predictions, _convert_simple_labels_to_torch_format(labels, device)
            )
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            # logging.info(f"Batch {batch_id} epoch {epoch} finished with loss {loss.item()}")

        logging.info(f"epoch {epoch} average loss: {epoch_loss / step:.4f}")
        logging.info(f"learning rate: {scheduler.get_last_lr()}")
        logging.info(f"epoch time: {time.time() - epoch_start_time}")
        training_losses.append(epoch_loss / step)

        if (epoch + 1) % validation_interval == 0:
            model.eval()
            with torch.no_grad():
                labels, predictions = get_model_predictions(
                    validation_data_loader, model, device
                )

                f_score = f1_score(
                    labels.cpu(),
                    predictions.cpu().argmax(dim=1),
                    average="weighted",
                )
                if f_score > best_f_score:
                    best_f_score = f_score
                loss_validation: torch.Tensor = loss_function(
                    predictions.to(torch.float64),
                    _convert_simple_labels_to_torch_format(labels, device),
                )

                if loss_validation < best_loss_validation:
                    if model_save_folder is not None:
                        trained_model_path = _save_model(
                            model, model_save_folder, optimizer, epoch
                        )
                        logging.info(f"Model saved at {trained_model_path}")
                    best_loss_validation = loss_validation
                logging.info(f"F1 score: {f_score}")
                logging.info(f"Validation loss: {loss_validation.data}")

                validation_losses.append(float(loss_validation.cpu().data))
        else:
            validation_losses.append(None)

    plot_training_curve(validation_losses, training_losses, visualizations_folder)

    trained_model_path = _save_model(
        model,
        model_save_folder
        if model_save_folder is not None
        else Config.saved_models_directory,
        optimizer,
        -1,
    )
    logging.info(
        f"Finished training, best f_score: {best_f_score}, best validation loss: {best_loss_validation.data}"
    )
    return best_f_score, trained_model_path


def _convert_simple_labels_to_torch_format(
    labels: torch.Tensor, device: torch.device
) -> torch.Tensor:
    return labels.to(torch.int64).to(device)


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
        labels = torch.cat([labels, validation_labels], dim=0)
    return labels, predictions


def _save_model(
    model: nn.Module,
    model_save_folder: Path,
    optimizer: Optimizer,
    epoch: int,
) -> Path:
    model_save_path = model_save_folder.joinpath(f"epoch_{epoch}.pth.tar")
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"Saving model at epoch: {epoch}")
    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        str(model_save_path),
    )
    return model_save_path

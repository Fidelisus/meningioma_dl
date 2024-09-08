import copy
import logging
from pathlib import Path
from typing import Optional, Dict, Callable
from typing import Tuple, Union, List

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from meningioma_dl.models.resnet import ResNet, set_model_parameters
from meningioma_dl.visualizations.images_visualization import visualize_images
from meningioma_dl.visualizations.results_visualizations import (
    plot_training_curve,
    TrainingMetrics,
)


def training_loop(
    training_data_loader: DataLoader,
    validation_data_loader: Optional[DataLoader],
    model: ResNet,
    optimizer: Optimizer,
    scheduler,
    loss_function,
    total_epochs: int,
    validation_interval: Optional[int],
    device: torch.device,
    evaluation_metric_weighting: str,
    visualizations_folder: Path = None,
    save_intermediate_models: bool = False,
    model_save_folder: Optional[Path] = None,
    logger: Callable[[str], None] = logging.info,
    save_images: bool = True,
    proximal_mu: Optional[float] = None,
) -> Tuple[float, Optional[Path], TrainingMetrics]:
    best_loss_validation = torch.tensor(np.inf)
    best_f_score = 0.0
    f_score = 0.0
    trained_model_path = None
    training_metrics = None
    logger(
        f"total_epochs: {total_epochs} batches_per_epoch: {len(training_data_loader)}"
    )
    best_model = copy.deepcopy(model).cpu()
    model = model.to(device)

    if proximal_mu is not None:
        global_params = copy.deepcopy(model)
    else:
        global_params = None

    training_losses: List[float] = []
    validation_losses: List[Union[float, None]] = []
    f_scores: List[Union[float, None]] = []
    learning_rates: List[Union[float, None]] = []
    for epoch in range(total_epochs):
        logger(f"Start epoch {epoch}")
        step = 0
        epoch_loss = 0

        model.train()

        for batch_id, batch_data in enumerate(training_data_loader):
            step += 1
            inputs, labels = batch_data["img"].to(device), batch_data["label"].to(
                device
            )
            if save_images and visualizations_folder is not None and epoch == 0:
                _save_batch_images(
                    batch_data, visualizations_folder.joinpath("training_images")
                )

            optimizer.zero_grad()
            predictions = model(inputs).to(torch.float64)

            labels_torch_format = _convert_simple_labels_to_torch_format(labels, device)
            if proximal_mu is None:
                loss = loss_function(predictions, labels_torch_format)
            else:
                loss = _calculate_loss_using_proximal_term(
                    global_params,
                    model,
                    labels_torch_format,
                    predictions,
                    loss_function,
                    proximal_mu,
                    logger,
                )

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        last_lr = scheduler.get_last_lr()
        learning_rates.append(last_lr[0])
        training_losses.append(epoch_loss / step)
        logger(
            f"Epoch {epoch} average loss: {epoch_loss / step:.4f}, "
            f"learning rate: {last_lr}"
        )

        if (
            validation_data_loader is not None
            and validation_interval is not None
            and (epoch + 1) % validation_interval == 0
        ):
            model.eval()
            with torch.no_grad():
                labels, predictions, _ = get_model_predictions(
                    validation_data_loader, model, device
                )
                loss_validation: torch.Tensor = loss_function(
                    predictions.to(torch.float64),
                    _convert_simple_labels_to_torch_format(labels, device),
                )
                if loss_validation < best_loss_validation:
                    best_loss_validation = loss_validation

                f_score = f1_score(
                    labels.cpu(),
                    predictions.cpu().argmax(dim=1),
                    average=evaluation_metric_weighting,
                )
                if f_score > best_f_score:
                    logger(
                        f"A better model obtained. new f_score: {f_score}, old: {f_score}"
                    )
                    best_model = copy.deepcopy(model).cpu()
                    if model_save_folder is not None:
                        best_f_score = f_score
                        if save_intermediate_models:
                            trained_model_path = _save_model(
                                model.state_dict(), model_save_folder, epoch
                            )
                        else:
                            trained_model_path = _save_model(
                                model.state_dict(),
                                model_save_folder,
                                -1,  # -1 used to override previous best model
                            )
                        logger(f"Model saved at {trained_model_path}")
                logger(f"F1 score: {f_score}, validation loss: {loss_validation.data}")

                validation_losses.append(float(loss_validation.cpu().data))
                f_scores.append(f_score)
        else:
            validation_losses.append(None)
            f_scores.append(None)
            best_model = copy.deepcopy(model).cpu()

        training_metrics = TrainingMetrics(
            validation_losses,
            training_losses,
            f_scores,
            learning_rates,
        )
        if visualizations_folder is not None:
            plot_training_curve(
                validation_losses,
                training_losses,
                f_scores,
                learning_rates,
                visualizations_folder,
            )
    set_model_parameters(model.cpu(), best_model.state_dict().values())
    logger(
        f"Finished training, last f_score: {f_score}, "
        f"best f_score: {best_f_score}, "
        f"best validation loss: {best_loss_validation.data}"
    )
    return best_f_score, trained_model_path, training_metrics


def _calculate_loss_using_proximal_term(
    global_params: ResNet,
    model: ResNet,
    labels_torch_format: torch.Tensor,
    predictions: torch.Tensor,
    loss_function: Callable,
    proximal_mu: float,
    logging_fn: Callable,
) -> torch.Tensor:
    proximal_term = 0.0
    for local_weights, global_weights in zip(
        model.parameters(), global_params.parameters()
    ):
        proximal_term += (local_weights - global_weights).norm(2)
    loss = (
        loss_function(predictions, labels_torch_format)
        + (proximal_mu / 2.0) * proximal_term
    )
    logging_fn(f"Loss calculated with a proximal term of {proximal_term}")
    return loss


def _save_batch_images(batch_data: Dict, directory: Path) -> None:
    visualize_images(
        batch_data["img"],
        directory,
        [
            f"{Path(file).stem.split('.')[0]}_label_{label.data+1}"
            for file, label in zip(batch_data["img_path"], batch_data["label"])
        ],
    )


def _convert_simple_labels_to_torch_format(
    labels: torch.Tensor, device: torch.device
) -> torch.Tensor:
    return labels.to(torch.int64).to(device)


def get_model_predictions(
    validation_data_loader: DataLoader, model: nn.Module, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, List[Path]]:
    predictions = torch.tensor([], dtype=torch.float32, device=device)
    labels = torch.tensor([], dtype=torch.long, device=device)
    images_paths: List[Path] = []
    for validation_data in validation_data_loader:
        validation_images, validation_labels = validation_data["img"].to(
            device
        ), validation_data["label"].to(device)
        predictions = torch.cat([predictions, model(validation_images)], dim=0)
        labels = torch.cat([labels, validation_labels], dim=0)
        images_paths.extend(Path(file) for file in validation_data["img_path"])
    return labels, predictions, images_paths


def _save_model(
    model,
    model_save_folder: Path,
    epoch: int,
) -> Path:
    model_save_path = model_save_folder.joinpath(f"epoch_{epoch}.pth.tar")
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"epoch": epoch, "state_dict": model},
        str(model_save_path),
    )
    return model_save_path

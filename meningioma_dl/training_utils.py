import logging
from pathlib import Path
from typing import Optional
from typing import Tuple, Union, List

import numpy as np
import torch
from monai.data import DataLoader
from sklearn.metrics import f1_score, cohen_kappa_score
from torch import nn, optim
from torch.optim.optimizer import Optimizer

from meningioma_dl.visualizations.results_visualizations import plot_training_curve

SCHEDULERS = {
    "exponent": optim.lr_scheduler.ExponentialLR,
    "cosine": optim.lr_scheduler.CosineAnnealingWarmRestarts,
}


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
    save_intermediate_models: bool = False,
    model_save_folder: Optional[Path] = None,
) -> Tuple[float, Optional[Path]]:
    best_loss_validation = torch.tensor(np.inf)
    best_f_score = 0.0
    batches_per_epoch = len(training_data_loader)
    logging.info(f"total_epochs: {total_epochs} batches_per_epoch: {batches_per_epoch}")
    # loss_function = WeightedKappaLoss(3)
    loss_function = nn.CrossEntropyLoss(
        weight=torch.tensor(loss_function_class_weights).to(torch.float64),
    )

    loss_function = loss_function.to(device)

    training_losses: List[float] = []
    validation_losses: List[Union[float, None]] = []
    f_scores: List[Union[float, None]] = []
    cohen_kappa_scores: List[Union[float, None]] = []
    for epoch in range(total_epochs):
        logging.info(f"Start epoch {epoch}")
        step = 0
        epoch_loss = 0

        model.train()

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
            epoch_loss += loss.item()
            # logging.info(f"Batch {batch_id} epoch {epoch} finished with loss {loss.item()}")
        scheduler.step()

        logging.info(
            f"Epoch {epoch} average loss: {epoch_loss / step:.4f}, "
            f"learning rate: {scheduler.get_last_lr()}"
        )
        training_losses.append(epoch_loss / step)

        if (epoch + 1) % validation_interval == 0:
            model.eval()
            with torch.no_grad():
                labels, predictions = get_model_predictions(
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
                    average="weighted",
                )
                cohen_kappa = cohen_kappa_score(
                    labels.cpu(),
                    predictions.cpu().argmax(dim=1),
                )
                cohen_kappa_scores.append(cohen_kappa)

                if f_score > best_f_score:
                    best_f_score = f_score
                    if save_intermediate_models:
                        trained_model_path = _save_model(
                            model, model_save_folder, optimizer, epoch
                        )
                    else:
                        trained_model_path = _save_model(
                            model,
                            model_save_folder,
                            optimizer,
                            -1,  # -1 used to override previous best model
                        )
                    logging.info(f"Model saved at {trained_model_path}")
                logging.info(
                    f"F1 score: {f_score}, cohen_kappa: {cohen_kappa_scores} validation loss: {loss_validation.data}"
                )

                validation_losses.append(float(loss_validation.cpu().data))
                f_scores.append(f_score)
        else:
            validation_losses.append(None)
            f_scores.append(None)
            cohen_kappa_scores.append(None)

    plot_training_curve(
        validation_losses,
        training_losses,
        f_scores,
        cohen_kappa_scores,
        visualizations_folder,
    )

    logging.info(
        f"Finished training, last f_score: {f_score}, "
        f"best f_score: {best_f_score}, "
        f"best validation loss: {best_loss_validation.data}"
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
    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        str(model_save_path),
    )
    return model_save_path


class WeightedKappaLoss(nn.Module):
    """
    Implements Weighted Kappa Loss. Weighted Kappa Loss was introduced in the
    [Weighted kappa loss function for multi-class classification
      of ordinal data in deep learning]
      (https://www.sciencedirect.com/science/article/abs/pii/S0167865517301666).
    Weighted Kappa is widely used in Ordinal Classification Problems. The loss
    value lies in $[-\infty, \log 2]$, where $\log 2$ means the random prediction
    Usage: loss_fn = WeightedKappaLoss(num_classes = NUM_CLASSES)
    """

    def __init__(
        self,
        num_classes: int,
        mode: Optional[str] = "quadratic",
        name: Optional[str] = "cohen_kappa_loss",
        epsilon: Optional[float] = 1e-10,
    ):
        """Creates a `WeightedKappaLoss` instance.
        Args:
          num_classes: Number of unique classes in your dataset.
          weightage: (Optional) Weighting to be considered for calculating
            kappa statistics. A valid value is one of
            ['linear', 'quadratic']. Defaults to 'quadratic'.
          name: (Optional) String name of the metric instance.
          epsilon: (Optional) increment to avoid log zero,
            so the loss will be $ \log(1 - k + \epsilon) $, where $ k $ lies
            in $ [-1, 1] $. Defaults to 1e-10.
        Raises:
          ValueError: If the value passed for `weightage` is invalid
            i.e. not any one of ['linear', 'quadratic']
        """

        super(WeightedKappaLoss, self).__init__()
        self.num_classes = num_classes
        if mode == "quadratic":
            self.y_pow = 2
        if mode == "linear":
            self.y_pow = 1

        self.epsilon = epsilon

    def kappa_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        device = y_pred.device
        y_pred = y_pred.to(torch.double)

        num_classes = self.num_classes
        y = torch.eye(num_classes).to(device)
        y_true = y[y_true].to(torch.double)

        repeat_op = (
            torch.Tensor(list(range(num_classes)))
            .unsqueeze(1)
            .repeat((1, num_classes))
            .to(device)
        )
        repeat_op_sq = torch.square((repeat_op - repeat_op.T))
        weights = repeat_op_sq / ((num_classes - 1) ** 2)

        pred_ = y_pred**self.y_pow
        pred_norm = pred_ / (self.epsilon + torch.reshape(torch.sum(pred_, 1), [-1, 1]))

        hist_rater_a = torch.sum(pred_norm, 0)
        hist_rater_b = torch.sum(y_true, 0)

        conf_mat = torch.matmul(pred_norm.T, y_true.to(torch.double))

        bsize = y_pred.size(0)
        nom = torch.sum(weights * conf_mat)
        expected_probs = torch.matmul(
            torch.reshape(hist_rater_a, [num_classes, 1]),
            torch.reshape(hist_rater_b, [1, num_classes]),
        )
        denom = torch.sum(weights * expected_probs / bsize)

        return nom / (denom + self.epsilon)

    def forward(self, y_pred, y_true):
        return self.kappa_loss(y_pred, y_true)

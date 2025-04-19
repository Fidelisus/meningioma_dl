from pathlib import Path
from typing import Tuple, List

import torch
from torch import nn
from torch.utils.data import DataLoader


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

from pathlib import Path
from typing import List, Optional

import monai
import torch
from monai.data import DataLoader
from monai.transforms import (
    Transform,
    LoadImaged,
    ScaleIntensityd,
    Resized,
    RandFlipd,
    RandRotate90d,
    Compose,
)

from meningioma_dl.data_loading.labels_loading import get_images_with_labels


def get_data_loader(
    labels_file_path: Path,
    data_root_directory: Path,
    num_workers: int,
    add_augmentation: bool,
    batch_size: Optional[int] = None,
) -> tuple[DataLoader, list[int]]:
    images, labels = get_images_with_labels(data_root_directory, labels_file_path)

    if batch_size is None:
        batch_size = len(labels)

    data_loader = init_data_loader(
        images,
        labels,
        batch_size=batch_size,
        num_workers=num_workers,
        add_augmentation=add_augmentation,
    )
    return data_loader, labels


def init_data_loader(
    images: List[str],
    labels: List[int],
    batch_size: int,
    num_workers: int,
    add_augmentation: bool = True,
) -> DataLoader:
    file_label_map = [
        {"img": img, "label": label} for img, label in zip(images, labels)
    ]

    base_transforms: list[Transform] = [
        LoadImaged(keys=["img"], ensure_channel_first=True, image_only=True),
        ScaleIntensityd(keys=["img"]),
        Resized(keys=["img"], spatial_size=(224, 224, 224)),
    ]

    augmentation_transforms: list[Transform] = [
        RandFlipd(keys=["img"], spatial_axis=0, prob=0.5),
        RandRotate90d(keys=["img"], prob=0.8, spatial_axes=(0, 2)),
    ]

    transformations = base_transforms
    if add_augmentation:
        transformations.extend(augmentation_transforms)

    dataset = monai.data.Dataset(
        data=file_label_map, transform=Compose(transformations)
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return data_loader

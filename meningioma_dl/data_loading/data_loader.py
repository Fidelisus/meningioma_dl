from typing import List

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


def get_data_loader(
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
        LoadImaged(keys=["img"], ensure_channel_first=True),
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

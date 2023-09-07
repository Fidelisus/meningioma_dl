import logging
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

import monai
import numpy as np
import torch
from monai.data import DataLoader
from monai import transforms

from meningioma_dl.data_loading.labels_loading import get_images_with_labels


"""
In order to make Cropforegroundd work you have to add the following code to 
venv1/lib/python3.7/site-packages/monai/transforms/croppad/array.py CropForeground crop_pad()
just after cropped = SpatialCrop(roi_start=box_start, roi_end=box_end)(img) and comment this line:

        # cropped = SpatialCrop(roi_start=box_start, roi_end=box_end)(img)
        # MY ADDITION START
        slices = tuple(
            slice(box_start[i], box_end[i])
            for i in range(img.ndim-1)
        )
        cropped=np.array([img[0][slices]])
        # MY ADDITION END
"""


class TransformationsMode(Enum):
    ONLY_LOAD = 0
    ONLY_PREPROCESSING = 1
    AUGMENT = 2


def get_data_loader(
    labels_file_path: Path,
    data_root_directory: Path,
    num_workers: int,
    transformations_mode: TransformationsMode = TransformationsMode.AUGMENT,
    batch_size: Optional[int] = None,
    augmentation_settings: Optional[List[transforms.Transform]] = None,
) -> Tuple[DataLoader, List[int]]:
    images, masks, labels = get_images_with_labels(
        data_root_directory, labels_file_path
    )

    if batch_size is None:
        batch_size = len(labels)

    data_loader = init_data_loader(
        images,
        masks,
        labels,
        batch_size=batch_size,
        num_workers=num_workers,
        transformations_mode=transformations_mode,
        augmentation_settings=augmentation_settings,
    )
    return data_loader, labels


def mask_image(data_dict):
    data_dict["image"] = data_dict["image"] * data_dict["label"]
    return data_dict


def init_data_loader(
    images: List[str],
    masks: List[str],
    labels: List[int],
    batch_size: int,
    num_workers: int,
    transformations_mode: TransformationsMode = TransformationsMode.AUGMENT,
    augmentation_settings: Optional[List[transforms.Transform]] = None,
) -> DataLoader:
    file_label_map = [
        {"img": img, "mask": mask, "label": label - 1}
        for img, label, mask in zip(images, labels, masks)
    ]

    transformations: List[transforms.Transform] = [
        transforms.LoadImaged(
            keys=["img", "mask"],
            meta_keys=["img_meta_dict", "mask_meta_dict"],
        ),
        transforms.EnsureChannelFirstd(
            keys=["img", "mask"], meta_keys=["img_meta_dict", "mask_meta_dict"]
        ),
        transforms.Orientationd(
            keys=["img", "mask"],
            meta_keys=["img_meta_dict", "mask_meta_dict"],
            axcodes="PLI",
        ),
        transforms.Spacingd(
            keys=["img", "mask"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
            meta_keys=["img_meta_dict", "mask_meta_dict"],
        ),
    ]

    if transformations_mode.value in {
        TransformationsMode.ONLY_PREPROCESSING.value,
        TransformationsMode.AUGMENT.value,
    }:
        transformations.extend(
            [
                transforms.CropForegroundd(keys=["img", "mask"], source_key="mask"),
                transforms.SpatialPadd(
                    keys=["img", "mask"],
                    spatial_size=(10, 10, 10),  # spatial_size=(151, 151, 151)
                ),
                transforms.Zoomd(keys=["mask"], zoom=1.2),
                transforms.MaskIntensityd(keys=["img"], mask_key="mask"),
                transforms.ScaleIntensityd(keys=["img"], minv=0.0, maxv=1.0),
            ]
        )

    if transformations_mode.value == TransformationsMode.AUGMENT.value:
        if augmentation_settings is None:
            logging.warning("No augmentation settings provided, using default ones")
            augmentation_settings: List[transforms.Transform] = [
                transforms.RandFlipd(keys=["img"], spatial_axis=0, prob=1),
                transforms.RandRotated(keys=["img"], prob=1),
                transforms.RandZoomd(keys=["img"], min_zoom=0.8, max_zoom=1.2, prob=1),
                transforms.RandGaussianNoised(keys=["img"], prob=1.0, std=0.2),
                # We need to mask after gaussian to avoid adding noise to the empty parts
                transforms.MaskIntensityd(keys=["img"], mask_key="mask"),
                transforms.Rand3DElasticd(
                    keys=["img"],
                    sigma_range=(0, 1),
                    magnitude_range=(3, 6),
                    prob=1.0,
                    rotate_range=(np.pi / 4),
                    padding_mode="zeros",
                ),
            ]
        transformations.extend(augmentation_settings)

    transformations.append(
        transforms.Resized(
            keys=["img", "mask"], spatial_size=(10, 10, 10)
        )  # spatial_size=(224, 224, 224))
    )

    dataset = monai.data.Dataset(
        data=file_label_map, transform=transforms.Compose(transformations)
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return data_loader

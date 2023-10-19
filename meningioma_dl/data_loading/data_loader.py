import logging
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

import monai
import torch
from monai import transforms
from monai.data import DataLoader

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


@dataclass
class PreprocessingSettings:
    initial_pad_spatial_size: int = 100
    final_resize_spatial_pad: Optional[int] = 224
    final_resize_mode: Optional[str] = "area"
    final_crop_and_pad_spatial_size: Optional[int] = None
    tissue_around_tumour_zoom: float = 1.2
    do_foreground_cropping: bool = True


def get_data_loader(
    labels_file_path: Path,
    data_root_directory: Path,
    num_workers: int,
    batch_size: int = 1,
    transformations_mode: TransformationsMode = TransformationsMode.AUGMENT,
    augmentation_settings: Optional[List[transforms.Transform]] = None,
    preprocessing_settings: PreprocessingSettings = PreprocessingSettings(),
) -> Tuple[DataLoader, List[int]]:
    images, masks, labels = get_images_with_labels(
        data_root_directory, labels_file_path
    )

    data_loader = init_data_loader(
        images,
        masks,
        labels,
        batch_size=batch_size,
        num_workers=num_workers,
        transformations_mode=transformations_mode,
        augmentation_settings=augmentation_settings,
        preprocessing_settings=preprocessing_settings,
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
    preprocessing_settings: PreprocessingSettings = PreprocessingSettings(),
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
                    spatial_size=preprocessing_settings.initial_pad_spatial_size,
                ),
                transforms.Zoomd(
                    keys=["mask"], zoom=preprocessing_settings.tissue_around_tumour_zoom
                ),
            ]
        )
        if preprocessing_settings.do_foreground_cropping:
            transformations.append(
                transforms.MaskIntensityd(keys=["img"], mask_key="mask")
            )

    if transformations_mode.value == TransformationsMode.AUGMENT.value:
        if augmentation_settings is None:
            logging.warning("No augmentation settings provided, using default ones")
            probability = 0.2
            augmentation_settings: List[transforms.Transform] = [
                transforms.RandFlipd(
                    keys=["img", "mask"], spatial_axis=0, prob=probability
                ),
                transforms.RandFlipd(
                    keys=["img", "mask"], spatial_axis=1, prob=probability
                ),
                transforms.RandFlipd(
                    keys=["img", "mask"], spatial_axis=2, prob=probability
                ),
                transforms.RandRotated(
                    keys=["img", "mask"],
                    prob=probability,
                    range_x=math.pi / 2,
                    range_y=math.pi / 2,
                    range_z=math.pi / 2,
                ),
                transforms.RandZoomd(
                    keys=["img", "mask"], min_zoom=0.8, max_zoom=1.2, prob=probability
                ),
                transforms.RandAffined(
                    keys=["img", "mask"],
                    translate_range=[(-10, 10), (-10, 10), (-10, 10)],
                    prob=probability,
                ),
                transforms.RandStdShiftIntensityd(
                    keys=["img"], factors=0.05, prob=probability
                ),
                transforms.RandGaussianNoised(keys=["img"], prob=probability, std=0.15),
                # transforms.Rand3DElasticd(
                #     keys=["img", "mask"],
                #     sigma_range=(0, 1),
                #     magnitude_range=(3, 6),
                #     prob=1.0,
                #     rotate_range=(np.pi / 4),
                #     padding_mode="zeros",
                # ),
            ]
        transformations.extend(augmentation_settings)
        if preprocessing_settings.do_foreground_cropping:
            transformations.append(
                # We need to mask after gaussian to avoid adding noise to the empty parts
                transforms.MaskIntensityd(keys=["img"], mask_key="mask")
            )

    if preprocessing_settings.final_resize_mode is not None:
        transformations.append(
            transforms.Resized(
                keys=["img", "mask"],
                spatial_size=preprocessing_settings.final_resize_spatial_pad,
                size_mode="longest",
                mode=preprocessing_settings.final_resize_mode,
            )
        )
        transformations.append(
            transforms.SpatialPadd(
                keys=["img", "mask"],
                spatial_size=preprocessing_settings.final_resize_spatial_pad,
            )
        )
    if preprocessing_settings.final_crop_and_pad_spatial_size is not None:
        transformations.append(
            transforms.SpatialPadd(
                keys=["img", "mask"],
                spatial_size=preprocessing_settings.final_crop_and_pad_spatial_size,
            )
        )
        transformations.append(
            transforms.CenterSpatialCropd(
                keys=["img", "mask"],
                roi_size=preprocessing_settings.final_crop_and_pad_spatial_size,
            )
        )

    transformations.append(transforms.ScaleIntensityd(keys=["img"], minv=0.0, maxv=1.0))

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

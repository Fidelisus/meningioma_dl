import logging
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple, Sequence, Dict, Any

import pandas as pd
import torch
from monai import transforms
from monai.data import Dataset
from monai.utils import set_determinism
from torch.utils.data import DataLoader, ConcatDataset

from meningioma_dl.data_loading.labels_loading import (
    get_images_with_labels,
)
from meningioma_dl.experiments_specs.preprocessing_specs import PreprocessingSpecs
from meningioma_dl.federated_learning.create_federated_data_splits import (
    get_uniform_client_partitions,
)


class TransformationsMode(Enum):
    ONLY_LOAD = 0
    ONLY_PREPROCESSING = 1
    AUGMENT = 2


class PadToLargestDimension:
    def __init__(self, key: str):
        self.key = key

    def __call__(self, data):
        max_dimension = 0
        for item in data[self.key]:
            max_shape = max(item.shape)
            if max_shape > max_dimension:
                max_dimension = max_shape
        return transforms.SpatialPadd(
            keys=[self.key],
            spatial_size=max_dimension,
        )(data)


def get_data_loader(
    labels_file_path: Path,
    data_root_directory: Path,
    batch_size: int = 1,
    transformations_mode: TransformationsMode = TransformationsMode.AUGMENT,
    augmentations: Optional[Sequence[transforms.Transform]] = None,
    preprocessing_specs: PreprocessingSpecs = PreprocessingSpecs(),
    class_mapping: Optional[Dict[int, int]] = None,
    client_specific_preprocessing: Optional[Dict[int, Optional[str]]] = None,
    manual_seed: int = 123,
) -> Tuple[DataLoader, List[int]]:
    images, masks, labels = get_images_with_labels(
        data_root_directory, labels_file_path, class_mapping
    )
    if client_specific_preprocessing is not None:
        samples_df = pd.DataFrame({"images": images, "masks": masks, "labels": labels})
        partitions = get_uniform_client_partitions(
            samples_df["labels"],
            len(client_specific_preprocessing.keys()),
            manual_seed,
        )
        datasets = get_client_specific_preprocessing_datasets(
            samples_df,
            partitions,
            client_specific_preprocessing,
            preprocessing_specs,
            augmentations,
            transformations_mode,
        )
        dataset = ConcatDataset(datasets.values())
    else:
        dataset = init_dataset(
            images,
            masks,
            labels,
            transformations_mode=transformations_mode,
            augmentations=augmentations,
            preprocessing_specs=preprocessing_specs,
        )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=torch.cuda.is_available(),
        shuffle=True,
    )
    return data_loader, labels


def init_dataset(
    images: List[str],
    masks: List[str],
    labels: List[int],
    transformations_mode: TransformationsMode = TransformationsMode.AUGMENT,
    augmentations: Optional[List[transforms.Transform]] = None,
    preprocessing_specs: PreprocessingSpecs = PreprocessingSpecs(),
    bounding_box_of_segmentation: int = 129,
) -> Dataset:
    set_determinism(seed=123)
    file_label_map = [
        {"img": img, "mask": mask, "label": label - 1, "img_path": img}
        for img, label, mask in zip(images, labels, masks)
    ]

    transformations: List[transforms.Transform] = [
        transforms.LoadImaged(keys=["img", "mask"]),
        transforms.EnsureChannelFirstd(keys=["img", "mask"]),
        transforms.Orientationd(keys=["img", "mask"], axcodes="PLI"),
        transforms.Spacingd(keys=["img", "mask"], pixdim=(1.0, 1.0, 1.0)),
    ]

    if transformations_mode.value in {
        TransformationsMode.ONLY_PREPROCESSING.value,
        TransformationsMode.AUGMENT.value,
    }:
        if not preprocessing_specs.do_foreground_cropping:
            cutting_size = (
                preprocessing_specs.initial_pad_spatial_size
                or bounding_box_of_segmentation
            )
            transformations.extend(
                [
                    transforms.CropForegroundd(
                        keys=["img"],
                        source_key="mask",
                        k_divisible=bounding_box_of_segmentation * 2,
                        allow_smaller=True,
                    ),
                    transforms.CropForegroundd(
                        keys=["mask"],
                        source_key="mask",
                    ),
                    transforms.SpatialPadd(
                        keys=["mask"],
                        spatial_size=cutting_size,
                    ),
                    PadToLargestDimension(key="mask"),
                    transforms.ShiftIntensityd(
                        keys=["mask"],
                        offset=1.0,
                    ),
                    transforms.SpatialPadd(
                        keys=["mask"],
                        spatial_size=bounding_box_of_segmentation * 2,
                    ),
                    transforms.CropForegroundd(
                        keys=["img", "mask"],
                        source_key="mask",
                    ),
                ]
            )
        else:
            transformations.extend(
                [
                    transforms.CropForegroundd(keys=["img", "mask"], source_key="mask"),
                    transforms.SpatialPadd(
                        keys=["img", "mask"],
                        spatial_size=preprocessing_specs.initial_pad_spatial_size,
                    ),
                    transforms.Zoomd(
                        keys=["mask"],
                        zoom=preprocessing_specs.tissue_around_tumour_zoom,
                    ),
                ]
            )
            if preprocessing_specs.histogram_shift_num_control_points is not None:
                transformations.append(
                    transforms.RandHistogramShiftd(
                        keys=["img"],
                        prob=1.0,
                        num_control_points=preprocessing_specs.histogram_shift_num_control_points,
                    )
                )
            if preprocessing_specs.bias_field_coeff is not None:
                transformations.append(
                    transforms.RandBiasFieldd(
                        keys=["img"],
                        prob=1.0,
                        coeff_range=(
                            preprocessing_specs.bias_field_coeff,
                            preprocessing_specs.bias_field_coeff + 0.000001,
                        ),
                    )
                )
            transformations.append(
                transforms.MaskIntensityd(keys=["img"], mask_key="mask")
            )
    if transformations_mode.value == TransformationsMode.AUGMENT.value:
        if augmentations is None:
            logging.warning("No augmentation settings provided, using no augmentations")
            augmentations = []
        transformations.extend(augmentations)
        if preprocessing_specs.do_foreground_cropping:
            transformations.append(
                # We need to mask after gaussian to avoid adding noise to the empty parts
                transforms.MaskIntensityd(keys=["img"], mask_key="mask")
            )

    if not transformations_mode.value == TransformationsMode.ONLY_LOAD.value:
        if preprocessing_specs.final_resize_mode is not None:
            transformations.append(
                transforms.Resized(
                    keys=["img", "mask"],
                    spatial_size=preprocessing_specs.final_resize_spatial_pad,
                    size_mode="longest",
                    mode=preprocessing_specs.final_resize_mode,
                )
            )
            transformations.append(
                transforms.SpatialPadd(
                    keys=["img", "mask"],
                    spatial_size=preprocessing_specs.final_resize_spatial_pad,
                )
            )
        if preprocessing_specs.final_crop_and_pad_spatial_size is not None:
            transformations.append(
                transforms.SpatialPadd(
                    keys=["img", "mask"],
                    spatial_size=preprocessing_specs.final_crop_and_pad_spatial_size,
                )
            )
            transformations.append(
                transforms.CenterSpatialCropd(
                    keys=["img", "mask"],
                    roi_size=preprocessing_specs.final_crop_and_pad_spatial_size,
                )
            )

        transformations.append(
            transforms.ScaleIntensityd(keys=["img"], minv=0.0, maxv=1.0)
        )

    dataset = Dataset(
        data=file_label_map, transform=transforms.Compose(transformations)
    )
    return dataset


def get_client_specific_preprocessing_datasets(
    samples_df: pd.DataFrame,
    partitions: Dict[int, Any],
    client_specific_preprocessings: Optional[Dict[int, Optional[str]]],
    default_preprocessing_specs: PreprocessingSpecs,
    augmentations: Optional[List[transforms.Transform]],
    transformations_mode: TransformationsMode,
) -> Dict[int, Dataset]:
    datasets = {}
    for client_id, indexes in partitions.items():
        preprocessing = default_preprocessing_specs
        client_specific_preprocessing = None
        if client_specific_preprocessings is not None:
            client_specific_preprocessing = client_specific_preprocessings[client_id]
            if client_specific_preprocessing is not None:
                preprocessing = PreprocessingSpecs.get_from_name(
                    client_specific_preprocessing
                )
        datasets[client_id] = init_dataset(
            samples_df.images.iloc[indexes].values,
            samples_df.masks.iloc[indexes].values,
            samples_df.labels.iloc[indexes].values,
            transformations_mode=transformations_mode,
            augmentations=augmentations,
            preprocessing_specs=preprocessing,
        )
        logging.info(
            f"Client: {client_id}, preprocessing: {client_specific_preprocessing}, "
            f"images: {[file[-26:-10] for file in samples_df.images.iloc[indexes].values]}"
        )
    return datasets

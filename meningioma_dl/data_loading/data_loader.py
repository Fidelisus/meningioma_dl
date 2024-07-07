import logging
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple, Sequence, Set, Dict

import monai
import torch
from monai import transforms
from monai.data import DataLoader
from monai.utils import set_determinism

from meningioma_dl.data_loading.labels_loading import get_images_with_labels
from meningioma_dl.experiments_specs.preprocessing_specs import PreprocessingSpecs

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
) -> Tuple[DataLoader, List[int]]:
    images, masks, labels = get_images_with_labels(
        data_root_directory, labels_file_path, class_mapping
    )

    data_loader = init_data_loader(
        images,
        masks,
        labels,
        batch_size=batch_size,
        transformations_mode=transformations_mode,
        augmentations=augmentations,
        preprocessing_specs=preprocessing_specs,
    )
    return data_loader, labels


def init_data_loader(
    images: List[str],
    masks: List[str],
    labels: List[int],
    batch_size: int,
    transformations_mode: TransformationsMode = TransformationsMode.AUGMENT,
    augmentations: Optional[List[transforms.Transform]] = None,
    preprocessing_specs: PreprocessingSpecs = PreprocessingSpecs(),
    bounding_box_of_segmentation: int = 129,
) -> DataLoader:
    set_determinism(seed=123)
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

    dataset = monai.data.Dataset(
        data=file_label_map, transform=transforms.Compose(transformations)
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=torch.cuda.is_available(),
        shuffle=True,
    )

    return data_loader

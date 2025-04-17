import math
from dataclasses import dataclass
from typing import List

from monai import transforms
from typing_extensions import Self


def get_basic_augmentation(
    probability_base: float, mask_after_gaussian: bool = True
) -> List[transforms.Transform]:
    affined_translate = 5
    transformations = [
        transforms.RandStdShiftIntensityd(
            keys=["img", "mask"],
            prob=probability_base + 0.05,
            factors=0.05,
        ),
        transforms.RandGaussianNoised(
            keys=["img"],
            std=0.1,
            prob=probability_base,
        ),
    ]
    if mask_after_gaussian:
        transformations.append(transforms.MaskIntensityd(keys=["img"], mask_key="mask"))
    transformations.extend(
        [
            transforms.RandRotated(
                keys=["img", "mask"],
                prob=probability_base + 0.1,
                range_x=math.pi / 4,
                range_y=math.pi / 4,
                range_z=math.pi / 4,
                padding_mode="zeros",
                keep_size=False,
            ),
            transforms.RandZoomd(
                keys=["img", "mask"],
                min_zoom=0.9,
                max_zoom=1.1,
                prob=probability_base + 0.05,
                keep_size=False,
            ),
            transforms.RandAffined(
                keys=["img", "mask"],
                prob=probability_base + 0.05,
                translate_range=[
                    (-affined_translate, affined_translate),
                    (-affined_translate, affined_translate),
                    (-affined_translate, affined_translate),
                ],
                padding_mode="zeros",
            ),
        ]
    )
    return transformations


def get_strong_augmentation(
    probability_base: float,
    mask_after_gaussian: bool = True,
    gaussian_std: float = 0.15,
) -> List[transforms.Transform]:
    affined_translate = 15
    transformations = [
        transforms.RandStdShiftIntensityd(
            keys=["img", "mask"],
            prob=probability_base + 0.05,
            factors=0.2,
        ),
        transforms.RandGaussianNoised(
            keys=["img"],
            std=gaussian_std,
            prob=probability_base,
        ),
    ]
    if mask_after_gaussian:
        transformations.append(transforms.MaskIntensityd(keys=["img"], mask_key="mask"))
    transformations.extend(
        [
            transforms.RandRotated(
                keys=["img", "mask"],
                prob=probability_base + 0.1,
                range_x=math.pi / 4,
                range_y=math.pi / 4,
                range_z=math.pi / 4,
                padding_mode="zeros",
                keep_size=False,
            ),
            transforms.RandZoomd(
                keys=["img", "mask"],
                min_zoom=0.8,
                max_zoom=1.2,
                prob=probability_base + 0.05,
                keep_size=False,
            ),
            transforms.RandAffined(
                keys=["img", "mask"],
                prob=probability_base + 0.05,
                translate_range=[
                    (-affined_translate, affined_translate),
                    (-affined_translate, affined_translate),
                    (-affined_translate, affined_translate),
                ],
                padding_mode="zeros",
            ),
        ]
    )
    return transformations


def get_augmentation_with_bias_correction_and_contrast(
    probability_base: float,
) -> List[transforms.Transform]:
    gaussian_std = 0.1
    affined_translate = 20
    transformations = [
        transforms.RandStdShiftIntensityd(
            keys=["img"],
            prob=probability_base,
            factors=0.2,
        ),
        transforms.RandGaussianNoised(
            keys=["img"],
            std=gaussian_std,
            prob=probability_base,
        ),
        transforms.RandAdjustContrastd(
            keys=["img"],
            prob=probability_base,
            gamma=(0.8, 1.5),
        ),
        # Great for MRI images, because bias field is often present in them
        transforms.RandBiasFieldd(
            keys=["img"],
            prob=probability_base,
            coeff_range=(0.0, 0.1),
        ),
        transforms.MaskIntensityd(keys=["img"], mask_key="mask"),
    ]
    transformations.extend(
        [
            transforms.RandRotated(
                keys=["img", "mask"],
                prob=probability_base,
                range_x=math.pi / 4,
                range_y=math.pi / 4,
                range_z=math.pi / 4,
                padding_mode="zeros",
                keep_size=False,
            ),
            transforms.RandZoomd(
                keys=["img", "mask"],
                min_zoom=0.8,
                max_zoom=1.2,
                prob=probability_base,
                keep_size=False,
            ),
            transforms.RandAffined(
                keys=["img", "mask"],
                prob=probability_base,
                translate_range=[
                    (-affined_translate, affined_translate),
                    (-affined_translate, affined_translate),
                    (-affined_translate, affined_translate),
                ],
                padding_mode="zeros",
            ),
        ]
    )
    return transformations


AUGMENTATIONS = {
    "no_augment": [],
    "basic_01p": get_basic_augmentation(0.1),
    "basic_02p": get_basic_augmentation(0.2),
    "basic_03p": get_basic_augmentation(0.3),
    "basic_05p": get_basic_augmentation(0.5),
    "basic_08p": get_basic_augmentation(0.8),
    "basic_1p": get_basic_augmentation(1.0),
    "strong_05p": get_strong_augmentation(0.5),
    "strong_08p": get_strong_augmentation(0.8),
    "with_bias_correction_05p": get_augmentation_with_bias_correction_and_contrast(0.5),
    "with_bias_correction_08p": get_augmentation_with_bias_correction_and_contrast(0.8),
    "with_bias_correction_1p": get_augmentation_with_bias_correction_and_contrast(1.0),
    "strong_less_gaussian_05p": get_strong_augmentation(0.5, gaussian_std=0.1),
    "strong_less_gaussian_08p": get_strong_augmentation(0.8, gaussian_std=0.1),
    "strong_less_gaussian_1p": get_strong_augmentation(1.0, gaussian_std=0.1),
    "basic_02p_no_mask_after_gaussian": get_basic_augmentation(0.2, False),
}


@dataclass
class AugmentationSpecs:
    name: str = "basic_05p"
    transformations_list: List[transforms.Transform] = ()

    @classmethod
    def get_from_name(cls, name: str) -> Self:
        return cls(name, AUGMENTATIONS[name])

    def __repr__(self):
        return f"Aumentations named: {self.name}"

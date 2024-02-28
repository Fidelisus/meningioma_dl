import math
from dataclasses import dataclass
from typing import List, Sequence

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


AUGMENTATIONS = {
    "no_augment": [],
    "basic_01p": get_basic_augmentation(0.1),
    "basic_02p": get_basic_augmentation(0.2),
    "basic_05p": get_basic_augmentation(0.5),
    "basic_08p": get_basic_augmentation(0.8),
    "basic_02p_no_mask_after_gaussian": get_basic_augmentation(0.2, False),
}


@dataclass
class AugmentationSpecs:
    name: str = "basic_05p"
    transformations_list: Sequence[transforms.Transform] = ()

    @classmethod
    def get_from_name(cls, name: str) -> Self:
        return cls(name, AUGMENTATIONS[name])

    def __repr__(self):
        return f"Aumentations named: {self.name}"

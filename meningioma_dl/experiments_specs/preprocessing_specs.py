from dataclasses import dataclass
from typing import Optional, Dict, Any

from typing_extensions import Self

PREPROCESSING_SPECS: Dict[str, Dict[str, Any]] = {
    "resize_mode_area": {"initial_pad_spatial_size": 50, "final_resize_mode": "area"},
    "resize_mode_nearest": {
        "final_resize_mode": "nearest",
        "initial_pad_spatial_size": 50,
    },
    "resize_mode_trilinear": {
        "final_resize_mode": "trilinear",
        "initial_pad_spatial_size": 50,
    },
    "30_padding": {"final_resize_mode": "trilinear", "initial_pad_spatial_size": 30},
    "50_padding": {"final_resize_mode": "trilinear", "initial_pad_spatial_size": 50},
    "70_padding": {"final_resize_mode": "trilinear", "initial_pad_spatial_size": 70},
    "100_padding": {"final_resize_mode": "trilinear", "initial_pad_spatial_size": 100},
    "no_padding": {"final_resize_mode": "trilinear", " initial_pad_spatial_size": 129},
    "no_resize": {"final_resize_mode": None, "final_crop_and_pad_spatial_size": 129},
    "no_resize_no_tissue_around_tumour": {
        "final_resize_mode": None,
        "final_crop_and_pad_spatial_size": 129,
        "tissue_around_tumour_zoom": None,
    },
    "no_resize_100_final_crop": {
        "final_resize_mode": None,
        "final_crop_and_pad_spatial_size": 100,
    },
    "no_resize_70_final_crop": {
        "final_resize_mode": None,
        "final_crop_and_pad_spatial_size": 70,
    },
    "no_0_foreground_30_padding": {
        "final_resize_mode": "trilinear",
        "initial_pad_spatial_size": 30,
        "do_foreground_cropping": False,
    },
    "no_0_foreground_70_padding": {
        "final_resize_mode": "trilinear",
        "initial_pad_spatial_size": 70,
        "do_foreground_cropping": False,
    },
    "no_0_foreground_no_padding": {
        "final_resize_mode": "trilinear",
        "do_foreground_cropping": False,
    },
}


@dataclass
class PreprocessingSpecs:
    initial_pad_spatial_size: int = 100
    final_resize_spatial_pad: Optional[int] = 224
    final_resize_mode: Optional[str] = "area"
    final_crop_and_pad_spatial_size: Optional[int] = None
    tissue_around_tumour_zoom: float = 1.2
    do_foreground_cropping: bool = True

    @classmethod
    def get_from_name(cls, name: str) -> Self:
        return cls(**PREPROCESSING_SPECS[name])

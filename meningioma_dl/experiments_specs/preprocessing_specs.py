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
    "30_padding": {
        "final_resize_mode": "trilinear",
        "initial_pad_spatial_size": 30,
        "final_resize_spatial_pad": 129,
    },
    "50_padding": {
        "final_resize_mode": "trilinear",
        "initial_pad_spatial_size": 50,
        "final_resize_spatial_pad": 129,
    },
    "70_padding": {
        "final_resize_mode": "trilinear",
        "initial_pad_spatial_size": 70,
        "final_resize_spatial_pad": 129,
    },
    "100_padding": {
        "final_resize_mode": "trilinear",
        "initial_pad_spatial_size": 100,
        "final_resize_spatial_pad": 129,
    },
    "no_padding": {"final_resize_mode": "trilinear", " initial_pad_spatial_size": 129},
    "no_resize": {},
    "histogram_shift_5": {"histogram_shift_num_control_points": 5},
    "histogram_shift_10": {"histogram_shift_num_control_points": 10},
    "histogram_shift_30": {"histogram_shift_num_control_points": 20},
    "histogram_shift_60": {"histogram_shift_num_control_points": 40},
    "bias_field_01": {"bias_field_coeff": 0.1},
    "bias_field_015": {"bias_field_coeff": 0.15},
    "bias_field_02": {"bias_field_coeff": 0.2},
    "bias_field_03": {"bias_field_coeff": 0.3},
    "bias_field_05": {"bias_field_coeff": 0.5},
    "bias_field_08": {"bias_field_coeff": 0.8},
    "bias_field_1": {"bias_field_coeff": 1.0},
    "bias_field_3": {"bias_field_coeff": 3.0},
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
    final_resize_mode: Optional[str] = None
    final_crop_and_pad_spatial_size: Optional[int] = 129
    tissue_around_tumour_zoom: float = 1.2
    do_foreground_cropping: bool = True
    histogram_shift_num_control_points: Optional[int] = None
    bias_field_coeff: Optional[float] = None

    @classmethod
    def get_from_name(cls, name: str) -> Self:
        return cls(**PREPROCESSING_SPECS[name])

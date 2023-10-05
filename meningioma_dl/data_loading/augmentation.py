import math
from dataclasses import dataclass
from typing import Any, Type, Dict, Tuple, List

from monai import transforms
from optuna import Trial


@dataclass
class HyperparameterSpecs:
    data_type: type


@dataclass
class Augmentation:
    transformation: Type[transforms.Transform]
    parameters: Dict[str, Any]
    hyperparameters: Dict[str, HyperparameterSpecs]


AUGMENTATIONS: Dict[str, Augmentation] = {
    "rand_flip_0_axis": Augmentation(
        transforms.RandFlipd,
        {"keys": ["img", "mask"], "spatial_axis": 0},
        {
            "prob": HyperparameterSpecs(data_type=float),
        },
    ),
    "rand_flip_1_axis": Augmentation(
        transforms.RandFlipd,
        {"keys": ["img", "mask"], "spatial_axis": 1},
        {
            "prob": HyperparameterSpecs(data_type=float),
        },
    ),
    "rand_flip_2_axis": Augmentation(
        transforms.RandFlipd,
        {"keys": ["img", "mask"], "spatial_axis": 2},
        {
            "prob": HyperparameterSpecs(data_type=float),
        },
    ),
    "rand_rotate": Augmentation(
        transforms.RandRotated,
        {
            "keys": ["img", "mask"],
        },
        {
            "prob": HyperparameterSpecs(data_type=float),
            "range_x": HyperparameterSpecs(data_type=float),
            "range_y": HyperparameterSpecs(data_type=float),
            "range_z": HyperparameterSpecs(data_type=float),
            "padding_mode": HyperparameterSpecs(data_type=str),
            "keep_size": HyperparameterSpecs(data_type=bool),
        },
    ),
    "rand_zoom": Augmentation(
        transforms.RandZoomd,
        {
            "keys": ["img", "mask"],
        },
        {
            "min_zoom": HyperparameterSpecs(data_type=float),
            "max_zoom": HyperparameterSpecs(data_type=float),
            "prob": HyperparameterSpecs(data_type=float),
            "keep_size": HyperparameterSpecs(data_type=bool),
        },
    ),
    "translate": Augmentation(
        transforms.RandAffined,
        {"keys": ["img", "mask"]},
        {
            "prob": HyperparameterSpecs(data_type=float),
            "translate_range": HyperparameterSpecs(data_type=float),
            "padding_mode": HyperparameterSpecs(data_type=str),
        },
    ),
    "shift_intensity": Augmentation(
        transforms.RandStdShiftIntensityd,
        {"keys": ["img"]},
        {
            "prob": HyperparameterSpecs(data_type=float),
            "factors": HyperparameterSpecs(data_type=float),
        },
    ),
    "gaussian_noise": Augmentation(
        transforms.RandGaussianNoised,
        {
            "keys": ["img"],
        },
        {
            "std": HyperparameterSpecs(data_type=float),
            "prob": HyperparameterSpecs(data_type=float),
        },
    ),
    "mask_after_gaussian": Augmentation(
        transforms.MaskIntensityd, {"keys": ["img"], "mask_key": "mask"}, {}
    ),
}


def suggest_hyperparameter_value(
    trial: Trial, name: str, values: Tuple, data_type: Type
) -> Any:
    if data_type == int:
        value = trial.suggest_int(name, *values)
    elif data_type == float:
        value = trial.suggest_float(name, *values)
    elif data_type == str:
        value = trial.suggest_categorical(name, values)
    else:
        raise ValueError("Sth went wrong with suggesting")

    return value


def propose_augmentation(
    trial: Trial, augmentation_settings: Dict[str, Dict[str, Tuple]]
) -> List[transforms.Transform]:
    augmentation_transforms: List[transforms.Transform] = []

    for augmentation_name, hyperparameters_dict in augmentation_settings.items():
        chosen_hyperparameters: Dict[str, Any] = {}
        for hyperparameter_name, hyperparameter_values in hyperparameters_dict.items():
            if type(hyperparameter_values) is tuple:
                chosen_hyperparameters[
                    hyperparameter_name
                ] = suggest_hyperparameter_value(
                    trial,
                    f"{augmentation_name}_{hyperparameter_name}",
                    hyperparameter_values,
                    AUGMENTATIONS[augmentation_name]
                    .hyperparameters[hyperparameter_name]
                    .data_type,
                )
            else:
                chosen_hyperparameters[hyperparameter_name] = hyperparameter_values
        augmentation_transforms.append(
            AUGMENTATIONS[augmentation_name].transformation(
                **AUGMENTATIONS[augmentation_name].parameters, **chosen_hyperparameters
            )
        )

    return augmentation_transforms

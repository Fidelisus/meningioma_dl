import math
from dataclasses import dataclass
from typing import Any, Type

from monai import transforms
from optuna import Trial


@dataclass
class HyperparameterSpecs:
    data_type: type


@dataclass
class Augmentation:
    transformation: Type[transforms.Transform]
    parameters: dict[str, Any]
    hyperparameters: dict[str, HyperparameterSpecs]


AUGMENTATIONS: dict[str, Augmentation] = {
    "rand_flip": Augmentation(
        transforms.RandFlipd,
        {
            "keys": ["img"],
        },
        {
            "spatial_axis": HyperparameterSpecs(data_type=int),
            "prob": HyperparameterSpecs(data_type=float),
        },
    ),
    "rand_rotate": Augmentation(
        transforms.RandRotated,
        {
            "keys": ["img"],
            "range_x": math.pi / 2,
            "range_y": math.pi / 2,
            "range_z": math.pi / 2,
        },
        {
            "prob": HyperparameterSpecs(data_type=float),
        },
    ),
    "rand_zoom": Augmentation(
        transforms.RandZoomd,
        {
            "keys": ["img"],
        },
        {
            "min_zoom": HyperparameterSpecs(data_type=float),
            "max_zoom": HyperparameterSpecs(data_type=float),
            "prob": HyperparameterSpecs(data_type=float),
        },
    ),
}

sample_augmentation_transforms: list[transforms.Transform] = [
    transforms.RandFlipd(keys=["img"], spatial_axis=0, prob=0.5),
    transforms.RandRotated(keys=["img"], prob=0.8),
    # gaussian noise
    # elastic deforamtions
]


def _suggest_hyperparameter_value(
    trial: Trial, name: str, values: tuple, data_type: type
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
    trial: Trial, augmentation_settings: dict[str, dict[str, tuple]]
) -> list[transforms.Transform]:
    augmentation_transforms: list[transforms.Transform] = []

    for augmentation_name, hyperparameters_dict in augmentation_settings.items():
        chosen_hyperparameters: dict[str, Any] = {}
        for hyperparameter_name, hyperparameter_values in hyperparameters_dict.items():
            if type(hyperparameter_values) is tuple:
                chosen_hyperparameters[
                    hyperparameter_name
                ] = _suggest_hyperparameter_value(
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

import math

from meningioma_dl.experiments_configs.data_types import (
    SearchSpace,
    HyperparametersConfig,
)


def get_default_augmentation(probability: float = 0.1) -> SearchSpace:
    return {
        "rand_flip_0_axis": {"prob": probability},
        "rand_flip_1_axis": {"prob": probability},
        "rand_flip_2_axis": {"prob": probability},
        "rand_rotate": {"prob": probability},
        "rand_zoom": {
            "min_zoom": 0.8,
            "max_zoom": 1.2,
            "prob": probability,
        },
        "translate": {
            "prob": probability,
        },
        "shift_intensity": {
            "factors": 0.05,
            "prob": probability,
        },
        "gaussian_noise": {
            "std": 0.15,
            "prob": probability,
        },
        "mask_after_gaussian": {},
    }


def get_exponential_learninig_rate(
    learning_rate: float,
    lr_scheduler_gamma: float,
) -> HyperparametersConfig:
    return {
        # TODO make it properly static
        "learning_rate": (learning_rate, learning_rate + 0.000001),
        "gamma": (lr_scheduler_gamma, lr_scheduler_gamma + 0.000001),
    }


def get_exp9_1_augmentation(probability: float = 0.1) -> SearchSpace:
    affined_translate = 5
    return {
        "shift_intensity": {
            "prob": probability + 0.05,
            "factors": 0.05,
        },
        "gaussian_noise": {
            "std": 0.1,
            "prob": probability,
        },
        "mask_after_gaussian": {},
        "rand_rotate": {
            "prob": probability + 0.1,
            "range_x": math.pi / 4,
            "range_y": math.pi / 4,
            "range_z": math.pi / 4,
            "padding_mode": "zeros",
            "keep_size": False,
        },
        "rand_zoom": {
            "min_zoom": 0.9,
            "max_zoom": 1.1,
            "prob": probability + 0.05,
            "keep_size": False,
        },
        "translate": {
            "prob": probability + 0.05,
            "translate_range": [
                (-affined_translate, affined_translate),
                (-affined_translate, affined_translate),
                (-affined_translate, affined_translate),
            ],
            "padding_mode": "zeros",
        },
    }


def get_exp9_2_augmentation(probability: float = 0.1) -> SearchSpace:
    affined_translate = 8
    return {
        "shift_intensity": {
            "prob": probability + 0.1,
            "factors": 0.1,
        },
        "gaussian_noise": {
            "std": 0.05,
            "prob": probability,
        },
        "mask_after_gaussian": {},
        "rand_rotate": {
            "prob": probability + 0.05,
            "range_x": math.pi / 4,
            "range_y": math.pi / 4,
            "range_z": math.pi / 4,
            "padding_mode": "zeros",
            "keep_size": False,
        },
        "rand_flip_1_axis": {"prob": probability + 0.05},
        "translate": {
            "prob": probability + 0.05,
            "translate_range": [
                (-affined_translate, affined_translate),
                (-affined_translate, affined_translate),
                (-affined_translate, affined_translate),
            ],
            "padding_mode": "zeros",
        },
    }


def get_exp9_3_augmentation(probability: float = 0.1) -> SearchSpace:
    affined_translate = 10
    return {
        "shift_intensity": {
            "prob": probability + 0.1,
            "factors": 0.1,
        },
        "gaussian_noise": {
            "std": 0.2,
            "prob": probability,
        },
        "mask_after_gaussian": {},
        "rand_rotate": {
            "prob": probability + 0.05,
            "range_x": math.pi / 4,
            "range_y": math.pi / 4,
            "range_z": math.pi / 4,
            "padding_mode": "zeros",
            "keep_size": False,
        },
        "rand_zoom": {
            "min_zoom": 0.8,
            "max_zoom": 1.2,
            "prob": probability,
            "keep_size": False,
        },
        "rand_flip_0_axis": {"prob": probability},
        "rand_flip_1_axis": {"prob": probability},
        "rand_flip_2_axis": {"prob": probability},
        "translate": {
            "prob": probability + 0.05,
            "translate_range": [
                (-affined_translate, affined_translate),
                (-affined_translate, affined_translate),
                (-affined_translate, affined_translate),
            ],
            "padding_mode": "zeros",
        },
    }


def get_cosine_learninig_rate(
    learning_rate: float,
    T_0: int,
    eta_min: float = 0.00001,
) -> HyperparametersConfig:
    return {
        "learning_rate": (learning_rate, learning_rate + 0.000001),
        "T_0": (T_0, T_0 + 1),
        "eta_min": (eta_min, eta_min + 0.000001),
    }

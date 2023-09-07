from numbers import Number
from typing import Tuple, Dict, Union

SearchSpace = Dict[str, Dict[str, Union[Tuple, Number]]]

FULL_SEARCH_SPACE_EXPERIMENT_1: SearchSpace = {
    "rand_flip": {"spatial_axis": (0, 1), "prob": 0.2},
    "rand_rotate": {"prob": 0.2},
    "rand_zoom": {
        "min_zoom": (0.8, 1.0),
        "max_zoom": (1.0, 1.2),
        "prob": 0.2,
    },
    "gaussian_noise": {
        "std": (0.01, 0.05),
        "prob": 0.2,
    },
    "mask_after_gaussian": {},
}

FULL_SEARCH_SPACE_EXPERIMENT_2: SearchSpace = {
    "rand_flip_0_axis": {"spatial_axis": 0, "prob": 0.2},
    "rand_flip_1_axis": {"spatial_axis": 1, "prob": 0.2},
    "rand_flip_2_axis": {"spatial_axis": 2, "prob": 0.2},
    "rand_rotate": {"prob": 0.3},
    "rand_zoom": {
        "min_zoom": 0.8,
        "max_zoom": 1.2,
        "prob": 0.3,
    },
    "translate": {
        "prob": 0.2,
    },
    "shift_intensity": {
        "prob": 0.2,
        "factors": 0.05,
    },
    "gaussian_noise": {
        "std": (0.05, 0.2),
        "prob": 0.3,
    },
    "mask_after_gaussian": {},
}


AFFINE_TRANSFORMS_SEARCH_SPACE: SearchSpace = {
    "rand_flip": {"spatial_axis": (0, 1), "prob": (0, 1)},
    "rand_rotate": {"prob": (0, 1)},
    "rand_zoom": {
        "min_zoom": (0.6, 1.0),
        "max_zoom": (1.0, 1.6),
        "prob": (0, 1),
    },
}

STATIC: SearchSpace = {
    "rand_flip": {"spatial_axis": 1, "prob": 0.5},
    "rand_rotate": {"prob": 0.5},
    "rand_zoom": {
        "min_zoom": 0.8,
        "max_zoom": 1.2,
        "prob": 0.5,
    },
    "gaussian_noise": {
        "std": 0.1,
        "prob": 0.5,
    },
    "mask_after_gaussian": {},
}

SEARCH_SPACES: Dict[str, SearchSpace] = {
    "full_exp_1": FULL_SEARCH_SPACE_EXPERIMENT_1,
    "full_exp_2": FULL_SEARCH_SPACE_EXPERIMENT_2,
    "affine_transforms": AFFINE_TRANSFORMS_SEARCH_SPACE,
    "static": STATIC,
}

HyperparametersConfig = Dict[str, Union[Tuple, Number]]

SIMPLE_LR_CONFIG_EXPERIMENT_1: HyperparametersConfig = {"learning_rate": (0.01, 0.5)}

SIMPLE_LR_CONFIG_EXPERIMENT_2: HyperparametersConfig = {"learning_rate": (0.0005, 0.1)}

HYPERPARAMETERS_CONFIGS: Dict[str, HyperparametersConfig] = {
    "simple_conf_exp_1": SIMPLE_LR_CONFIG_EXPERIMENT_1,
    "simple_conf_exp_2": SIMPLE_LR_CONFIG_EXPERIMENT_2,
}

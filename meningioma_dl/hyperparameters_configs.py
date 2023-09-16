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

FULL_SEARCH_SPACE_EXPERIMENT_3: SearchSpace = {
    "rand_flip_0_axis": {"prob": 0.1},
    "rand_flip_1_axis": {"prob": 0.1},
    "rand_flip_2_axis": {"prob": 0.1},
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
        "std": (0.02, 0.2),
        "prob": 0.3,
    },
    "mask_after_gaussian": {},
}


AFFINE_TRANSFORMS_SEARCH_SPACE: SearchSpace = {
    "rand_flip_0_axis": {"prob": (0, 1)},
    "rand_rotate": {"prob": (0, 1)},
    "rand_zoom": {
        "min_zoom": (0.6, 1.0),
        "max_zoom": (1.0, 1.6),
        "prob": (0, 1),
    },
}

STATIC: SearchSpace = {
    "rand_flip_0_axis": {"prob": 0.1},
    "rand_flip_1_axis": {"prob": 0.1},
    "rand_flip_2_axis": {"prob": 0.1},
    "rand_rotate": {"prob": 0.3},
    "rand_zoom": {
        "min_zoom": 0.8,
        "max_zoom": 1.2,
        "prob": 0.4,
    },
    "translate": {
        "prob": 0.1,
    },
    "shift_intensity": {
        "prob": 0.2,
        "factors": 0.05,
    },
    "gaussian_noise": {
        "std": 0.15,
        "prob": 0.3,
    },
    "mask_after_gaussian": {},
}

STATIC_EXPERIMENT_5: SearchSpace = {
    "rand_flip_0_axis": {"prob": 0.2},
    "rand_flip_1_axis": {"prob": 0.2},
    "rand_flip_2_axis": {"prob": 0.2},
    "rand_rotate": {"prob": 0.4},
    "rand_zoom": {
        "min_zoom": 0.8,
        "max_zoom": 1.2,
        "prob": 0.4,
    },
    "translate": {
        "prob": 0.3,
    },
    "shift_intensity": {
        "prob": 0.3,
        "factors": 0.05,
    },
    "gaussian_noise": {
        "std": 0.15,
        "prob": 0.4,
    },
    "mask_after_gaussian": {},
}

probability = 0.1
EXPERIMENT_8_AUGMENT_LOW_PROBABILITY: SearchSpace = {
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
        "prob": 0.3,
        "factors": probability,
    },
    "gaussian_noise": {
        "std": 0.15,
        "prob": probability,
    },
    "mask_after_gaussian": {},
}

SEARCH_SPACES: Dict[str, SearchSpace] = {
    "full_exp_1": FULL_SEARCH_SPACE_EXPERIMENT_1,
    "full_exp_3": FULL_SEARCH_SPACE_EXPERIMENT_3,
    "affine_transforms": AFFINE_TRANSFORMS_SEARCH_SPACE,
    "static": STATIC,
    "static_exp_5": STATIC_EXPERIMENT_5,
    "static_exp_6": STATIC_EXPERIMENT_5, # same config as exp 5
    "empty": {},
    "exp_8": EXPERIMENT_8_AUGMENT_LOW_PROBABILITY
}

HyperparametersConfig = Dict[str, Union[Tuple, Number]]

SIMPLE_LR_CONFIG_EXPERIMENT_1: HyperparametersConfig = {"learning_rate": (0.0005, 0.1)}

SIMPLE_LR_CONFIG_EXPERIMENT_3: HyperparametersConfig = {"learning_rate": (0.02, 0.3)}

STATIC_CONFIG: HyperparametersConfig = {"learning_rate": (0.005, 0.005001)}

STATIC_CONFIG_SCHEDULERS: HyperparametersConfig = {
    "learning_rate": (0.0005, 0.0005001),
    # "sgd_momentum": (0.99, 0.990001),
    # "weight_decay": (0.01, 0.01001),
    "lr_scheduler_gamma": (0.99, 0.99001),
}

ADAM_CONFIG_EXPERIMENT_5: HyperparametersConfig = {
    "learning_rate": (0.01, 0.3),
    # "sgd_momentum": (0.99, 0.990001),
    # "weight_decay": (0.01, 0.01001),
    "lr_scheduler_gamma": (0.8, 0.99),
}

ADAM_CONFIG_EXPERIMENT_6: HyperparametersConfig = {
    "learning_rate": (0.01, 0.1),
    # "sgd_momentum": (0.99, 0.990001),
    # "weight_decay": (0.01, 0.01001),
    "lr_scheduler_gamma": (0.95, 0.999),
}

HYPERPARAMETERS_CONFIGS: Dict[str, HyperparametersConfig] = {
    "simple_conf_exp_1": SIMPLE_LR_CONFIG_EXPERIMENT_1,
    "simple_conf_exp_3": SIMPLE_LR_CONFIG_EXPERIMENT_3,
    "static": STATIC_CONFIG,
    "static_schedulers": STATIC_CONFIG_SCHEDULERS,
    "adam_exp_5": ADAM_CONFIG_EXPERIMENT_5,
    "adam_exp_6": ADAM_CONFIG_EXPERIMENT_6,
}

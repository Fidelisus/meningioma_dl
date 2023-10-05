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


def get_learninig_rate_static_config(
    learning_rate: float,
    lr_scheduler_gamma: float,
) -> HyperparametersConfig:
    return {
        # TODO make it properly static
        "learning_rate": (learning_rate, learning_rate + 0.000001),
        "lr_scheduler_gamma": (lr_scheduler_gamma, lr_scheduler_gamma + 0.000001),
    }

from typing import Dict

from meningioma_dl.experiments_configs.data_types import (
    SearchSpace,
    HyperparametersConfig,
)
from meningioma_dl.experiments_configs.hyperparameters_configs import (
    FULL_SEARCH_SPACE_EXPERIMENT_1,
    FULL_SEARCH_SPACE_EXPERIMENT_3,
    AFFINE_TRANSFORMS_SEARCH_SPACE,
    STATIC,
    STATIC_EXPERIMENT_5,
    SIMPLE_LR_CONFIG_EXPERIMENT_1,
    SIMPLE_LR_CONFIG_EXPERIMENT_3,
    STATIC_CONFIG,
    STATIC_CONFIG_SCHEDULERS,
    ADAM_CONFIG_EXPERIMENT_5,
    ADAM_CONFIG_EXPERIMENT_6,
)
from meningioma_dl.experiments_configs.hyperparameters_configs_v2 import (
    get_default_augmentation,
    get_learninig_rate_static_config,
)

AUGMENTATIONS_SEARCH_SPACES: Dict[str, SearchSpace] = {
    "full_exp_1": FULL_SEARCH_SPACE_EXPERIMENT_1,
    "full_exp_3": FULL_SEARCH_SPACE_EXPERIMENT_3,
    "affine_transforms": AFFINE_TRANSFORMS_SEARCH_SPACE,
    "static": STATIC,
    "static_exp_5": STATIC_EXPERIMENT_5,
    "static_exp_6": STATIC_EXPERIMENT_5,  # same config as exp 5
    "empty": {},
    "exp_8_01_augment_prob": get_default_augmentation(0.1),
    "exp_8_02_augment_prob": get_default_augmentation(0.2),
    "exp_8_005_augment_prob": get_default_augmentation(0.05),
}
HYPERPARAMETERS_CONFIGS: Dict[str, HyperparametersConfig] = {
    "simple_conf_exp_1": SIMPLE_LR_CONFIG_EXPERIMENT_1,
    "simple_conf_exp_3": SIMPLE_LR_CONFIG_EXPERIMENT_3,
    "static": STATIC_CONFIG,
    "static_schedulers": STATIC_CONFIG_SCHEDULERS,
    "adam_exp_5": ADAM_CONFIG_EXPERIMENT_5,
    "adam_exp_6": ADAM_CONFIG_EXPERIMENT_6,
    "0001_lr_099_gamma": get_learninig_rate_static_config(0.001, 0.99),
    "00005_lr_099_gamma": get_learninig_rate_static_config(0.0005, 0.99),
    "0002_lr_09_gamma": get_learninig_rate_static_config(0.002, 0.9),
}

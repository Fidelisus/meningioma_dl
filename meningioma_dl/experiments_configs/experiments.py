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
    get_exponential_learninig_rate,
    get_exp9_1_augmentation,
)

AUGMENTATIONS_SEARCH_SPACES: Dict[str, SearchSpace] = {
    "full_exp_1": FULL_SEARCH_SPACE_EXPERIMENT_1,
    "full_exp_3": FULL_SEARCH_SPACE_EXPERIMENT_3,
    "affine_transforms": AFFINE_TRANSFORMS_SEARCH_SPACE,
    "static": STATIC,
    "static_exp_5": STATIC_EXPERIMENT_5,
    "static_exp_6": STATIC_EXPERIMENT_5,  # same config as exp 5
    "empty": {},
    "exp_8_005_augment_prob": get_default_augmentation(0.05),
    "exp_8_01_augment_prob": get_default_augmentation(0.1),
    "exp_8_02_augment_prob": get_default_augmentation(0.2),
    "exp_9_1_005_augment_prob": get_exp9_1_augmentation(0.05),
    "exp_9_1_01_augment_prob": get_exp9_1_augmentation(0.1),
    "exp_9_1_03_augment_prob": get_exp9_1_augmentation(0.3),
}

HYPERPARAMETERS_CONFIGS: Dict[str, HyperparametersConfig] = {
    "simple_conf_exp_1": SIMPLE_LR_CONFIG_EXPERIMENT_1,
    "simple_conf_exp_3": SIMPLE_LR_CONFIG_EXPERIMENT_3,
    "static": STATIC_CONFIG,
    "static_schedulers": STATIC_CONFIG_SCHEDULERS,
    "adam_exp_5": ADAM_CONFIG_EXPERIMENT_5,
    "adam_exp_6": ADAM_CONFIG_EXPERIMENT_6,
    "00005_lr_0999_gamma": get_exponential_learninig_rate(0.0005, 0.999),
    "0001_lr_099_gamma": get_exponential_learninig_rate(0.001, 0.999),
    "0001_lr_0999_gamma": get_exponential_learninig_rate(0.001, 0.99),
    "0002_lr_09_gamma": get_exponential_learninig_rate(0.002, 0.9),
    "0002_lr_0999_gamma": get_exponential_learninig_rate(0.002, 0.999),
    "0002_lr_099_gamma": get_exponential_learninig_rate(0.002, 0.99),
    "0003_lr_099_gamma": get_exponential_learninig_rate(0.003, 0.99),
    "0005_lr_099_gamma": get_exponential_learninig_rate(0.005, 0.99),
    "001_lr_099_gamma": get_exponential_learninig_rate(0.01, 0.99),
    "002_lr_09_gamma": get_exponential_learninig_rate(0.02, 0.9),
}

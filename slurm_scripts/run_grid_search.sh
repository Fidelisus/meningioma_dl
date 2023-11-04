#!/bin/bash

slurm_script_path=/home/cir/lsobocinski/meningioma_dl/slurm_scripts/centralized_training.sh
# augmentation_configs=("exp_9_1_01_augment_prob" "exp_9_1_025_augment_prob")
# hyperparameters_configs=("0001_lr_0999_gamma" "002_lr_099_gamma" "0005_lr_0995_gamma")
# hyperparameters_configs=("cosine_lr_0003_t0_20" "cosine_lr_0002_t0_40")

augmentation_configs=("no_augmentation")
# hyperparameters_configs=("0001_lr_0999_gamma" "002_lr_099_gamma" "0005_lr_0995_gamma")

hyperparameters_configs=("1_lr_099_gamma")
scheduler_name="exponent"
preprocessing_settings=("no_padding")
resnet_layers_to_unfreeze=(1 2 3 4)
use_training_data_for_validation="True"
loss_function_name="cross_entropy"

runs_main_name="unfreeze_${resnet_layers_to_unfreeze}"

for augmentation_config in "${augmentation_configs[@]}"; do
    for hyperparameters_config in "${hyperparameters_configs[@]}"; do
        for preprocessing_setting in "${preprocessing_settings[@]}"; do
            for layer in "${resnet_layers_to_unfreeze[@]}"; do
                sbatch -p full --qos jobarray "$slurm_script_path" "$augmentation_config" \
                "$hyperparameters_config" "$runs_main_name" "$scheduler_name" "$preprocessing_setting" \
                "$layer" "$use_training_data_for_validation" "$loss_function_name"
                sleep 1m
            done
        done
    done
done

#!/bin/bash

slurm_script_path=/home/cir/lsobocinski/meningioma_dl/slurm_scripts/centralized_training.sh
augmentation_configs=("exp_9_000_augment_prob" "exp_9_005_augment_prob" "exp_9_01_augment_prob")
hyperparameters_configs=("0001_lr_0999_gamma" "0002_lr_0999_gamma" "0002_lr_099_gamma" "0003_lr_099_gamma")
runs_main_name=also_augment_mask_gut_feeling_augment_run_2

for augmentation_config in "${augmentation_configs[@]}"; do
    for hyperparameters_config in "${hyperparameters_configs[@]}"; do
        sbatch -p full $slurm_script_path $augmentation_config $hyperparameters_config $runs_main_name
        sleep 1m
    done
done

#!/bin/bash

slurm_script_path=/home/cir/lsobocinski/meningioma_dl/slurm_scripts/centralized_training.sh

augmentation_specs=("basic_01p")
scheduler_specs=("05_lr_099_gamma")
preprocessing_specs=("no_resize")
training_settings=("central_200_epochs")

runs_main_name="1"

for augmentation in "${augmentation_specs[@]}"; do
    for scheduler in "${scheduler_specs[@]}"; do
        for preprocessing in "${preprocessing_specs[@]}"; do
            for training in "${training_settings[@]}"; do
                sbatch -p full --qos jobarray "$slurm_script_path" \
                  "$augmentation" \
                  "$scheduler" \
                  "$preprocessing" \
                  "$training" \
                  "$runs_main_name"
                sleep 1m
            done
        done
    done
done

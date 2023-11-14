#!/bin/bash

slurm_script_path=/home/cir/lsobocinski/meningioma_dl/slurm_scripts/centralized_training.sh

augmentation_specs=("basic_05p")
scheduler_specs=("1_lr_099_gamma")
preprocessing_specs=("no_resize")
training_settings=("central_200_epochs")
model_specs=("resnet_10_1_unfreezed" "resnet_10_4_unfreezed")

runs_main_name="3"

for augmentation in "${augmentation_specs[@]}"; do
    for scheduler in "${scheduler_specs[@]}"; do
        for preprocessing in "${preprocessing_specs[@]}"; do
            for training in "${training_settings[@]}"; do
                for model in "${model_specs[@]}"; do
                    sbatch -p full --qos jobarray "$slurm_script_path" \
                    "$augmentation" \
                    "$scheduler" \
                    "$preprocessing" \
                    "$training" \
                    "$model" \
                    "$runs_main_name"
                    sleep 30s
                done
            done
        done
    done
done

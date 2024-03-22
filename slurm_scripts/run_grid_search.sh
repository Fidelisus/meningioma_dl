#!/bin/bash

slurm_script_path=/home/cir/lsobocinski/meningioma_dl/slurm_scripts/run_training.sh

augmentation_specs=("basic_05p")
scheduler_specs=("01_lr_099_gamma")
preprocessing_specs=("no_resize")
training_settings=("federated_20r_20e_5c" "federated_10r_40e_5c" "federated_40r_10e_5c") #("federated_20r_20e_5c")
model_specs=("resnet_10_3_unfreezed")
fl_strategy_specs=("fed_avg_default")

#script_name="run_grid_search.py"

script_name="run_federated_training.py"

runs_main_name="FL0"

for augmentation in "${augmentation_specs[@]}"; do
    for scheduler in "${scheduler_specs[@]}"; do
        for preprocessing in "${preprocessing_specs[@]}"; do
            for training in "${training_settings[@]}"; do
                for model in "${model_specs[@]}"; do
                    for fl_strategy in "${fl_strategy_specs[@]}"; do
                      sbatch -p centos7 --qos jobarray \
                      "$slurm_script_path" \
                      "$augmentation" \
                      "$scheduler" \
                      "$preprocessing" \
                      "$training" \
                      "$model" \
                      "$fl_strategy" \
                      "$runs_main_name" \
                      "$script_name"
                      sleep 30s
                    done
                done
            done
        done
    done
done

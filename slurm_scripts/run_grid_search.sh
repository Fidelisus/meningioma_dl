#!/bin/bash

slurm_script_path=/home/cir/lsobocinski/meningioma_dl/slurm_scripts/run_training.sh

augmentation_specs=("basic_01p")
scheduler_specs=("001_lr_099_gamma" "05_lr_099_gamma")
preprocessing_specs=("no_resize")
# training_settings=("federated_40r_5e_3c" "federated_80r_5e_3c")
training_settings=("central_300_epochs")
model_specs=("class_2_and_3_together_3_unfreezed")
# fl_strategy_specs=("fed_avg_05_fraction")
fl_strategy_specs=("centralized")
seed=123

script_name="run_optuna_study.py"
# script_name="run_federated_training.py"

runs_main_name="fl1"

# I can also try --exclude instead of --nodelist

for augmentation in "${augmentation_specs[@]}"; do
    for scheduler in "${scheduler_specs[@]}"; do
        for preprocessing in "${preprocessing_specs[@]}"; do
            for training in "${training_settings[@]}"; do
                for model in "${model_specs[@]}"; do
                    for fl_strategy in "${fl_strategy_specs[@]}"; do
                      sbatch -p centos7 --nodelist=on4  --qos jobarray \
                      "$slurm_script_path" \
                      "$augmentation" \
                      "$scheduler" \
                      "$preprocessing" \
                      "$training" \
                      "$model" \
                      "$fl_strategy" \
                      "$runs_main_name" \
                      "$script_name" \
                      "$seed"
                      sleep 30s
                    done
                done
            done
        done
    done
done

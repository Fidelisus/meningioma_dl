#!/bin/bash

slurm_script_path=/home/cir/lsobocinski/meningioma_dl/slurm_scripts/run_training.sh

augmentation_specs=("with_bias_correction_08p")
scheduler_specs=("0001_lr_099_gamma")
preprocessing_specs=("no_resize")
# preprocessing_specs=("70_padding")
seed=123
cv_folds=(0 1 2 3 4)

# model_specs=("class_2_and_3_together_4_unfreezed")
model_specs=("resnet_10_4_unfreezed")

fl_strategy_specs=("centralized")
training_settings=("central_300_epochs")
script_name="run_optuna_study.py"
runs_main_name="cv_final_model"

# fl_strategy_specs=("fed_avg_05_fraction")
# training_settings=("federated_80r_5e_3c")
# script_name="run_federated_training.py"
# runs_main_name="f1"

# I can also try --exclude instead of --nodelist
node="on5"

# Those 2 need to be also here to avoid strange errors
module add "Python/3.9.5-GCCcore-8.2.0"
module add "PyTorch/1.9.0-foss-2019a"

for augmentation in "${augmentation_specs[@]}"; do
    for scheduler in "${scheduler_specs[@]}"; do
        for preprocessing in "${preprocessing_specs[@]}"; do
            for training in "${training_settings[@]}"; do
                for model in "${model_specs[@]}"; do
                    for fl_strategy in "${fl_strategy_specs[@]}"; do
                        for cv_fold in "${cv_folds[@]}"; do
                            sbatch -p centos7 --nodelist=$node --qos jobarray \
                            "$slurm_script_path" \
                            "$augmentation" \
                            "$scheduler" \
                            "$preprocessing" \
                            "$training" \
                            "$model" \
                            "$fl_strategy" \
                            "$runs_main_name" \
                            "$script_name" \
                            "$seed" \
                            "$cv_fold"
                            sleep 30s
                        done
                    done
                done
            done
        done
    done
done

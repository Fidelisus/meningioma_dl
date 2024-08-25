#!/bin/bash

slurm_script_path=/home/cir/lsobocinski/meningioma_dl/slurm_scripts/run_training.sh

augmentation_specs=("with_bias_correction_1p")
scheduler_specs=("001_lr_099_gamma")
preprocessing_specs=("no_resize")
# preprocessing_specs=("70_padding")
seed=123
cv_folds=(0 1 2 3)

model_specs=("class_2_and_3_together_4_unfreezed")
# model_specs=("resnet_10_4_unfreezed")


# fl_strategy_specs=("centralized")
# training_settings=("central_100_epochs" "central_100_epochs" "central_100_epochs")
# script_name="run_centralized_training.py"
# runs_main_name="cv_final_model"

fl_strategy_specs=("fed_avg")
training_settings=("federated_200r_1e_3c") # "histogram_shifts_low_200r_1e_3c" ks04_200r_1e_3c federated_200r_1e_3c histogram_shifts_high_200r_1e_3c 
script_name="run_federated_training.py"
runs_main_name="10_runs_cv"

# I can also try --exclude instead of --nodelist
node="on1"

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
                            sleep 60s
                        done
                    done
                done
            done
        done
    done
done

#!/bin/sh

#SBATCH --nodes=1

# GPU
#SBATCH --gres=gpu:1

#Email notification
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=e12123563@student.tuwien.ac.at

#SBATCH --ntasks=1
#SBATCH --mem=16384
#SBATCH --time=48:00:00
#SBATCH --job-name=meningioma_classification

base_dir=/home/cir/lsobocinski
venv_path=${base_dir}/meningioma_dl/venv1

n_workers=1
n_epochs=100

n_trials=1
augmentation_config="${1:-exp_8_005_augment_prob}"
hyperparameters_config_name="${2:-0002_lr_09_gamma}"
scheduler_name="${4:-exponent}"
preprocessing_settings_name="${5:-default}"
study_name="${3:-playground}_${augmentation_config}_${hyperparameters_config_name}_${preprocessing_settings_name}"
resnet_layers_to_unfreeze="${6:-0}"
use_training_data_for_validation="${7:-False}"
loss_function_name="${8:-cross_entropy}"

module add Python/3.7.3-foss-2019a
module add PyTorch/1.6.0-foss-2019a-Python-3.7.3
# virtualenv --system-site-packages ${base_dir}/meningioma_dl/venv
source ${venv_path}/bin/activate

echo "Running Slurm job with id $SLURM_JOBID and name ${study_name}_${SLURM_JOBID} "
echo "venv path: $venv_path"

#lddpython is needed to load a newer glibc
${base_dir}/meningioma_dl/slurm_scripts/lddpython ${base_dir}/meningioma_dl/meningioma_dl/run_optuna_study.py \
  --device_name="cuda" \
  --n_workers=${n_workers} \
  --env_file_path=${base_dir}/meningioma_dl/envs/slurm.env --n_epochs="${n_epochs}" \
  --n_trials="${n_trials}" --study_name="${study_name}" --run_id="${SLURM_JOBID}" \
  --batch_size=4 --validation_interval=4 --search_space_name="${augmentation_config}" \
  --hyperparameters_config_name="${hyperparameters_config_name}" \
  --scheduler_name="${scheduler_name}" \
  --preprocessing_settings_name="${preprocessing_settings_name}" \
  --resnet_layers_to_unfreeze="${resnet_layers_to_unfreeze}" \
  --use_training_data_for_validation="${use_training_data_for_validation}" \
  --loss_function_name="${loss_function_name}"

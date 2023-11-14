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

n_trials=1
augmentation_specs="${1:-basic_01p}"
scheduler_specs="${2:-05_lr_099_gamma}"
preprocessing_specs="${3:-no_resize}"
training_specs="${4:-central_2_epochs}"
model_specs="${5:-resnet_10_2_unfreezed}"
run_id="${6:-playground}_${preprocessing_specs}_${augmentation_specs}_${scheduler_specs}_${model_specs}_${SLURM_JOBID}"

module add Python/3.7.3-foss-2019a
module add PyTorch/1.6.0-foss-2019a-Python-3.7.3
# virtualenv --system-site-packages ${base_dir}/meningioma_dl/venv
source ${venv_path}/bin/activate

echo "Running Slurm job with id $SLURM_JOBID and name $run_id"
echo "venv path: $venv_path"

#lddpython is needed to load a newer glibc
${base_dir}/meningioma_dl/slurm_scripts/lddpython ${base_dir}/meningioma_dl/meningioma_dl/run_optuna_study.py \
  --device_name="cuda" \
  --env_file_path=${base_dir}/meningioma_dl/envs/slurm.env \
  --n_trials=$n_trials \
  --run_id="${run_id}" \
  --validation_interval=1 \
  --augmentations_specs_name="${augmentation_specs}" \
  --scheduler_specs_name="${scheduler_specs}" \
  --preprocessing_specs_name="${preprocessing_specs}" \
  --model_specs_name="${model_specs}" \
  --training_specs_name="${training_specs}"

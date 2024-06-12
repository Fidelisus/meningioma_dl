#!/bin/sh

#SBATCH --nodes=1

# GPU
#SBATCH --gres=gpu:1

#SBATCH --ntasks=1
#SBATCH --mem=16384
#SBATCH --time=0:15:00
#SBATCH --job-name=meningioma_classification

base_dir=/home/cir/lsobocinski
venv_path=${base_dir}/meningioma_dl/venv39

training_run_id="$1"
use_test_data="$2"
cv_fold="$3"
trained_model_path="${base_dir}/data/meningioma/optuna/trials/models/$training_run_id/epoch_-1.pth.tar"
# preprocessing_specs="70_padding"
preprocessing_specs="no_resize"
# model_specs="class_2_and_3_together_3_unfreezed"
model_specs="resnet_10_4_unfreezed"
run_id="eval_${training_run_id}_${SLURM_JOBID}"

module add "PyTorch/1.9.0-foss-2019a"
source ${venv_path}/bin/activate

echo "Running Slurm job with id $SLURM_JOBID and name $run_id"
echo "venv path: $venv_path"

#lddpython is needed to load a newer glibc
${base_dir}/meningioma_dl/slurm_scripts/lddpython ${base_dir}/meningioma_dl/meningioma_dl/evaluate.py \
  --trained_model_path="$trained_model_path" \
  --env_file_path=${base_dir}/meningioma_dl/envs/slurm.env \
  --run_id="${run_id}" \
  --device_name="cuda" \
  --preprocessing_specs_name="${preprocessing_specs}" \
  --model_specs_name="${model_specs}" --use_test_data="${use_test_data}" \
  --cv_fold="$cv_fold"

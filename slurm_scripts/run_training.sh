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
# venv_path=${base_dir}/meningioma_dl/venv1
venv_path=${base_dir}/meningioma_dl/venv39

augmentation_specs="${1:-basic_01p}"
scheduler_specs="${2:-05_lr_099_gamma}"
preprocessing_specs="${3:-no_resize}"
training_specs="${4:-central_2_epochs}"
model_specs="${5:-resnet_10_2_unfreezed}"
fl_strategy_specs="${6:-fed_avg_default}"
run_id="${7:-playground}_${preprocessing_specs}_${augmentation_specs}_${scheduler_specs}_${model_specs}_${SLURM_JOBID}"
script_name="${8:-run_grid_search.py}"

module add Python/3.9.5-GCCcore-8.2.0
module add PyTorch/1.9.0-foss-2019a
# module add Python/3.7.3-foss-2019a
# module add PyTorch/1.6.0-foss-2019a-Python-3.7.3
# virtualenv --system-site-packages ${venv_path}
source ${venv_path}/bin/activate

echo "Running Slurm job with id $SLURM_JOBID and name $run_id"
echo "venv path: $venv_path"
python -m pip freeze

#lddpython is needed to load a newer glibc
${base_dir}/meningioma_dl/slurm_scripts/lddpython ${base_dir}/meningioma_dl/meningioma_dl/"$script_name" \
  --device_name="cuda" \
  --env_file_path=${base_dir}/meningioma_dl/envs/ci_run_cluster.env \
  --run_id="${run_id}" \
  --augmentations_specs_name="${augmentation_specs}" \
  --scheduler_specs_name="${scheduler_specs}" \
  --preprocessing_specs_name="${preprocessing_specs}" \
  --model_specs_name="${model_specs}" \
  --training_specs_name="${training_specs}" \
  --fl_strategy_specs_name="${fl_strategy_specs}"

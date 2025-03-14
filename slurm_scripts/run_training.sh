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

base_dir=/home/cir/lsobocinski/meningioma_dl/

augmentation_specs="${1:-basic_01p}"
scheduler_specs="${2:-05_lr_099_gamma}"
preprocessing_specs="${3:-no_resize}"
training_specs="${4:-central_2_epochs}"
model_specs="${5:-resnet_10_2_unfreezed}"
fl_strategy_specs="${6:-fed_avg_default}"
script_name="${8:-run_centralized_training.py}"
seed="${9:-123}"
cv_fold="${10:-None}"

run_id="${7:-playground}_${fl_strategy_specs}_${training_specs}_${preprocessing_specs}_${augmentation_specs}_${scheduler_specs}_${model_specs}_fold${cv_fold}_${SLURM_JOBID}"

module add cuDNN/7.6.5.32-CUDA-10.1.243 > /dev/null 2>&1

echo "Running Slurm job with id $SLURM_JOBID and name $run_id"

singularity exec --nv ${base_dir}image.sif ${base_dir}entrypoint.sh ${base_dir}/meningioma_dl/"$script_name" \
  --device_name="cuda" \
  --env_file_path=${base_dir}/envs/slurm.env \
  --run_id="${run_id}" \
  --augmentations_specs_name="${augmentation_specs}" \
  --scheduler_specs_name="${scheduler_specs}" \
  --preprocessing_specs_name="${preprocessing_specs}" \
  --model_specs_name="${model_specs}" \
  --training_specs_name="${training_specs}" \
  --fl_strategy_specs_name="${fl_strategy_specs}" \
  --manual_seed="${seed}" \
  --cv_fold="${cv_fold}"

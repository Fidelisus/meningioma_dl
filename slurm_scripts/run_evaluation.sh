#!/bin/sh

#SBATCH --nodes=1

# GPU
#SBATCH --gres=gpu:1

#SBATCH --ntasks=1
#SBATCH --mem=8192
#SBATCH --time=0:30:00
#SBATCH --job-name=meningioma_classification

base_dir=/home/cir/lsobocinski/meningioma_dl/

training_run_id="$1"
use_test_data="$2"
cv_fold="$3"
ensemble_id="$4"
if [ "$ensemble_id" = "False" ]; then
   trained_model_path="/home/cir/lsobocinski/data/meningioma/optuna/trials/models/$training_run_id/epoch_-1.pth.tar"
   script_name="evaluate.py"
else
   trained_model_path="$ensemble_id"
   script_name="evaluate_ensemble.py"
fi
echo "Trained model path: $trained_model_path"

preprocessing_specs="no_resize"
model_specs="class_2_and_3_together_4_unfreezed"
#model_specs="resnet_10_4_unfreezed"
run_id="eval_${training_run_id}_${SLURM_JOBID}"

module add cuDNN/7.6.5.32-CUDA-10.1.243 > /dev/null 2>&1

echo "Running Slurm job with id $SLURM_JOBID and name $run_id"

singularity exec --nv ${base_dir}image.sif ${base_dir}entrypoint.sh ${base_dir}meningioma_dl/"$script_name" \
  "$trained_model_path" \
  --env_file_path=${base_dir}/envs/slurm.env \
  --run_id="${run_id}" \
  --device_name="cuda" \
  --preprocessing_specs_name="${preprocessing_specs}" \
  --model_specs_name="${model_specs}" --use_test_data="${use_test_data}" \
  --cv_fold="$cv_fold"

#!/bin/sh

#SBATCH --nodes=1

# GPU
#SBATCH --gres=gpu:1

#Email notification
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=e12123563@student.tuwien.ac.at

#SBATCH --ntasks=21
#SBATCH --mem=1000
#SBATCH --time=2:00:00
#SBATCH --job-name=gputest_lukasz

base_dir=/home/cir/lsobocinski
venv_path=${base_dir}/meningioma_dl/venv

#just add Tensorflow, as Tensorflow includes the other requirements (CUDA, cuDNN...)
# module add Torch/20170724-goolf-1.4.10
module add Python/3.6.2-goolf-1.4.10
virtualenv --system-site-packages ${base_dir}/meningioma_dl/venv
source ${venv_path}/bin/activate

echo "Running Slurm job with id $SLURM_JOBID"
echo "venv path: $venv_path"

#lddpython is needed to load a newer glibc
lddpython_ ${base_dir}/meningioma_dl/meningioma_dl/run_optuna_study.py \
  --env_file_path=${base_dir}/meningioma_dl/envs/slurm.env --n_epochs=2 \
  --n_trials=2 --study_name=playground_1 --run_id="$SLURM_JOBID"

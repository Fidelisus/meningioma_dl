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

num_workers=4
n_epochs=2
n_trials=2
study_name="playground_1"

module add Python/3.6.2-goolf-1.4.10
virtualenv --system-site-packages ${base_dir}/meningioma_dl/venv
source ${venv_path}/bin/activate

echo "Running Slurm job with id $SLURM_JOBID"
echo "venv path: $venv_path"

#lddpython is needed to load a newer glibc
lddpython_ ${base_dir}/meningioma_dl/meningioma_dl/run_optuna_study.py \
  --device_name="cuda" \
  --num_workers=${num_workers} \
  --env_file_path=${base_dir}/meningioma_dl/envs/slurm.env --n_epochs=${n_epochs} \
  --n_trials=${n_trials} --study_name=${study_name} --run_id="$SLURM_JOBID"

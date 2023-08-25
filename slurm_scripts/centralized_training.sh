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

#just add Tensorflow, as Tensorflow includes the other requirements (CUDA, cuDNN...)
module add Tensorflow

#lddpython is needed to load a newer glibc
lddpython cifar10_train.py --train_dir /tmp/cifar10_train_$SLURM_JOBID

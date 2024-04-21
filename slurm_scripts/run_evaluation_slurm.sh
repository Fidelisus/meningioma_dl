sbatch -p centos7 --nodelist=on3 --qos jobarray \
 /home/cir/lsobocinski/meningioma_dl/slurm_scripts/run_evaluation.sh \
 "$1"
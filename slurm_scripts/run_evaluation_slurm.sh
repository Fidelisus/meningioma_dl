use_test_data="True"

module add "Python/3.9.5-GCCcore-8.2.0"
module add "PyTorch/1.9.0-foss-2019a"

# --nodelist=on5
sbatch -p centos7 --qos normal  \
 /home/cir/lsobocinski/meningioma_dl/slurm_scripts/run_evaluation.sh "$1" "$use_test_data"
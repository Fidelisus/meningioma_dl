use_test_data="True"
fold=4
ensemble_id="False"

module add "Python/3.9.5-GCCcore-8.2.0"
module add "PyTorch/1.9.0-foss-2019a"

# --nodelist=on5
for run_id in "$@"
do
    sbatch -p centos7 --qos jobarray  \
 /home/cir/lsobocinski/meningioma_dl/slurm_scripts/run_evaluation.sh "$run_id" "$use_test_data" "$fold" "$ensemble_id"
done

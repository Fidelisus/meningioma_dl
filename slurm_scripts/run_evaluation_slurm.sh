use_test_data="False"

sbatch -p centos7 --nodelist=on3 --qos normal \
 /home/cir/lsobocinski/meningioma_dl/slurm_scripts/run_evaluation.sh "$1" "$use_test_data"
use_test_data="True"
fold=4
ensemble_id="False" # iid_fold_4a
weights_folder="None"
ensemble_weighting="None"

# --nodelist=on5
for run_id in "$@"
do
    sbatch -p centos7 --qos jobarray  \
 /home/cir/lsobocinski/meningioma_dl/slurm_scripts/run_evaluation.sh "$run_id" "$use_test_data" "$fold" "$ensemble_id" "$weights_folder" "$ensemble_weighting"
done

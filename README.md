# Development environment

1. Create a Python 3.13 venv

2. Install the project with required dependencies using:

```
pip install -e .
```

3. Please refer to the `jobs` module for the available jobs.

# Running the code as a slurm job 

Singularity, a library similar to Docker, was used to build and ship the containers.

```shell
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
```
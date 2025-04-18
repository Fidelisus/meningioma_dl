# Development environment

1. Create a Python 3.13 venv

2. Install the project with required dependencies using:

```shell
pip install -e .
```

3. Please refer to the `jobs` module for the available jobs.

4. To run any job, the following env variables need to be set:

```shell
# TODO
```

# Running the code as a slurm job 

Singularity, a library similar to Docker, was used to build and ship the containers.

1. Build a singularity container using `image.def`. Please refer to the official docs of Singularity on how to do it.

2. Execute the container. Please note that the directory with the code needs to be mounted to the container at `BASE_DIR`. Then, the library will be installed using `pip install .` in the `entrypoint.sh` script. 

```shell
BASE_DIR="XXX"
SCRIPT_NAME="run_centralized_training.py"
ENV_FILE_PATH="XXX"

singularity exec --nv ${BASE_DIR}/image.sif ${BASE_DIR}/entrypoint.sh ${BASE_DIR}/meningioma_dl/"${SCRIPT_NAME}" \
  --device_name="cuda" \
  --env_file_path=${ENV_FILE_PATH} \
  --run_id="test_run_123" \
  --augmentations_specs_name="basic_08p" \
  --model_specs_name="resnet_10_2_unfreezed" \
  --training_specs_name="central_2_epochs" \
  --manual_seed="123" \
  --cv_fold="0"
```
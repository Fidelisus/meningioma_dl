import json
from pathlib import Path
from typing import List, Dict

import torch

from meningioma_dl.experiments_specs.model_specs import ModelSpecs
from meningioma_dl.models.resnet import ResNet, load_model_from_file


def load_models(trained_model_paths: List[str], model_specs: ModelSpecs):
    models: List[ResNet] = []
    for trained_model_path in trained_model_paths:
        model = load_model_from_file(
            trained_model_path, model_specs, torch.device("cpu")
        )
        models.append(model)
    return models


def load_json(weights_folder: Path, file_name: str) -> Dict:
    with open(weights_folder.joinpath(file_name), "r") as f:
        return json.load(f)

import logging
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments_specs.model_specs import ModelSpecs
from model_evaluation.metrics import calculate_basic_metrics
from model_training.predictions import get_model_predictions
from models.resnet import ResNet, load_model_from_file


class EnsembleWeighting(Enum):
    GLOBAL_WEIGHTING = "global_weighting"
    LOCAL_WEIGHTING = "local_weighting"


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    return weights / weights.sum()


def local_models_vs_clients_f_scores_to_matrix(
    local_models_vs_clients_f_scores: Dict[int, Dict[int, float]],
) -> np.ndarray:
    local_models_vs_clients_f_scores_matrix = np.array(
        [
            list(local_models_vs_clients_f_scores[local_model_id].values())
            for local_model_id in sorted(local_models_vs_clients_f_scores.keys())
        ]
    )
    return local_models_vs_clients_f_scores_matrix


def get_local_ensemble_weights(
    local_models_vs_clients_f_scores: Dict[int, Dict[int, float]],
) -> Dict[int, Dict[int, float]]:
    local_models_vs_clients_f_scores_matrix = (
        local_models_vs_clients_f_scores_to_matrix(local_models_vs_clients_f_scores)
    )
    mean_weight_for_each_model: Dict[int, Dict[int, float]] = {}
    for client_id in local_models_vs_clients_f_scores:
        mean_weight_for_each_model_vector: np.ndarray = normalize_weights(
            local_models_vs_clients_f_scores_matrix[:, int(client_id)]
        )
        mean_weight_for_each_model[client_id] = {
            client_id: mean_weight
            for client_id, mean_weight in enumerate(mean_weight_for_each_model_vector)
        }
    return mean_weight_for_each_model


def get_global_ensemble_weights(
    local_models_vs_clients_f_scores: Dict[int, Dict[int, float]],
) -> Dict[int, float]:
    local_models_vs_clients_f_scores_matrix = (
        local_models_vs_clients_f_scores_to_matrix(local_models_vs_clients_f_scores)
    )
    mean_weight_for_each_model_vector: np.ndarray = normalize_weights(
        np.mean(local_models_vs_clients_f_scores_matrix, axis=1)
    )
    mean_weight_for_each_model = {
        client_id: mean_weight
        for client_id, mean_weight in enumerate(mean_weight_for_each_model_vector)
    }
    return mean_weight_for_each_model


def ensemble_weights_to_numpy(global_ensemble_weights: Dict[int, float]) -> np.ndarray:
    return np.array(list(global_ensemble_weights.values()))


def get_local_models_vs_clients_f_scores(
    validation_data_loaders: dict[int, DataLoader],
    model_specs: ModelSpecs,
    saved_models_folder: Path,
    device: torch.device,
    number_of_clients: int,
) -> Tuple[Dict[int, ResNet], Dict[int, Dict[int, float]]]:
    clients_models: Dict[int, ResNet] = {}
    local_models_vs_clients_f_scores: Dict[int, Dict[int, float]] = defaultdict(dict)
    for client_id in range(number_of_clients):
        client_model = load_model_from_file(
            trained_model_path=saved_models_folder.joinpath(
                f"best_model_{client_id}.pth.tar"
            ),
            model_specs=model_specs,
            device=torch.device("cpu"),
        )
        clients_models[client_id] = client_model
        for client_id_to_validate_on in range(number_of_clients):
            data_loader = validation_data_loaders[client_id_to_validate_on]
            client_model.eval()
            client_model = client_model.to(device)
            with torch.no_grad():
                labels, predictions, images_paths = get_model_predictions(
                    data_loader, client_model, device
                )
            predictions_flat = predictions.cpu().argmax(dim=1)
            f_score = calculate_basic_metrics(
                labels_cpu=labels.cpu(),
                predictions_flat=predictions_flat,
                evaluation_metric_weighting=model_specs.evaluation_metric_weighting,
                logger=logging.info,
            )
            local_models_vs_clients_f_scores[client_id][
                client_id_to_validate_on
            ] = f_score
    return clients_models, local_models_vs_clients_f_scores

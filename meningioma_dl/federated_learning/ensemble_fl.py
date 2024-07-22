from typing import Dict

import numpy as np


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    return weights / weights.sum()


def local_models_vs_clients_f_scores_to_matrix(
    local_models_vs_clients_f_scores: Dict[int, Dict[int, float]]
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

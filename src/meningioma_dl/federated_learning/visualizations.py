from pathlib import Path
from typing import List, Tuple

import numpy as np
from flwr.common import Metrics

from meningioma_dl.visualizations.results_visualizations import (
    deserialize_series,
    deserialize_value,
    plot_fl_training_curve,
)


def visualize_federated_learning_metrics(
    training_metrics: List[Tuple[int, Metrics]],
    validation_metrics: List[Tuple[int, Metrics]],
    visualizations_folder: Path,
    epochs_in_one_round: int,
):
    n_samples_per_client_training = []
    n_samples_per_client_validation = []
    validation_losses = []
    training_losses = []
    f_scores = []
    learning_rates = []
    for epoch_id, training_metrics_tuples in enumerate(training_metrics):
        n_samples_for_clients: List[int] = []
        losses_for_clients: List[np.ndarray] = []
        lr_for_clients: List[np.ndarray] = []
        for n_samples, metrics in training_metrics_tuples:
            n_samples_for_clients.append(n_samples)
            losses_for_clients.append(
                np.atleast_1d(deserialize_series(metrics, "training_losses"))
            )
            lr_for_clients.append(
                np.atleast_1d(deserialize_series(metrics, "learning_rates"))
            )
        n_samples_per_client_training.append(n_samples_for_clients)
        training_losses.append(np.hstack([losses_for_clients]))
        learning_rates.append(np.hstack([lr_for_clients]))

    for epoch_id, validation_metrics_tuples in enumerate(validation_metrics):
        n_samples_for_clients: List[int] = []
        losses_for_clients: List[np.ndarray] = []
        f_scores_for_clients: List[np.ndarray] = []
        for n_samples, metrics in validation_metrics_tuples:
            n_samples_for_clients.append(n_samples)
            nans_vector = np.full(epochs_in_one_round - 1, np.nan)
            losses_for_clients.append(
                np.concatenate([nans_vector, [deserialize_value(metrics, "loss")]])
            )
            f_scores_for_clients.append(
                np.concatenate([nans_vector, [deserialize_value(metrics, "f_score")]])
            )
        n_samples_per_client_validation.append(n_samples_for_clients)
        validation_losses.append(np.hstack([losses_for_clients]))
        f_scores.append(np.hstack([f_scores_for_clients]))

    plot_fl_training_curve(
        np.array(n_samples_per_client_training),
        np.array(n_samples_per_client_validation),
        np.array(validation_losses),
        np.array(training_losses),
        np.array(f_scores),
        np.array(learning_rates),
        visualizations_folder,
    )
    return {}

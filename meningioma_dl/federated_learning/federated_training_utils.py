from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple, Any, Dict, Optional, Sequence, Callable

import numpy as np
import pandas as pd
import torch
from flwr.common import Metrics
from monai import transforms
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from meningioma_dl.config import Config
from meningioma_dl.data_loading.data_loader import TransformationsMode, init_data_loader
from meningioma_dl.data_loading.labels_loading import get_images_with_labels
from meningioma_dl.experiments_specs.fl_strategy_specs import FLStrategySpecs
from meningioma_dl.experiments_specs.modelling_specs import ModellingSpecs
from meningioma_dl.experiments_specs.preprocessing_specs import PreprocessingSpecs
from meningioma_dl.experiments_specs.training_specs import FederatedTrainingSpecs
from meningioma_dl.federated_learning.server import SaveModelFedAvg
from meningioma_dl.models.resnet import ResNet
from meningioma_dl.utils import get_loss_function_class_weights
from meningioma_dl.visualizations.results_visualizations import plot_fl_training_curve


def get_optimizer_and_scheduler(
    parameters_to_fine_tune: List[torch.Tensor], modelling_specs: ModellingSpecs
) -> Tuple[torch.optim.Optimizer, Any]:
    lr_params = [
        {
            "params": parameters_to_fine_tune,
            "lr": modelling_specs.scheduler_specs.learning_rate,
        }
    ]
    optimizer = torch.optim.Adam(lr_params)
    scheduler = modelling_specs.scheduler_specs.get_scheduler(optimizer)
    return optimizer, scheduler


def get_data_loaders(
    modelling_specs: ModellingSpecs, training_specs: FederatedTrainingSpecs
) -> Tuple[Dict[int, DataLoader], Dict[int, DataLoader], Optional[torch.Tensor]]:
    training_data_loader, labels_train = get_federated_data_loaders(
        labels_file_path=Config.train_labels_file_path,
        data_root_directory=Config.data_directory,
        transformations_mode=TransformationsMode.AUGMENT,
        training_specs=training_specs,
        augmentations=modelling_specs.augmentation_specs.transformations_list,
        preprocessing_specs=modelling_specs.preprocessing_specs,
        class_mapping=modelling_specs.model_specs.class_mapping,
    )
    validation_data_loader, labels_validation = get_federated_data_loaders(
        labels_file_path=Config.validation_labels_file_path,
        data_root_directory=Config.data_directory,
        transformations_mode=TransformationsMode.ONLY_PREPROCESSING,
        training_specs=training_specs,
        preprocessing_specs=modelling_specs.preprocessing_specs,
        class_mapping=modelling_specs.model_specs.class_mapping,
    )
    loss_function_weighting = (
        torch.tensor(
            get_loss_function_class_weights(labels_train + labels_validation)
        ).to(torch.float64)
        if modelling_specs.model_specs.evaluation_metric_weighting == "weighted"
        else None
    )
    return training_data_loader, validation_data_loader, loss_function_weighting


def get_partitions(
    partitioning_mode: str, n_partitions: int, samples_df: pd.DataFrame
) -> Dict[int, Tuple[int, ...]]:
    partition_configs = {}
    if partitioning_mode == "uniform":
        # for i in range(n_partitions):
        #     partition_configs[i] = samples_df.index[:10]
        splitter = StratifiedKFold(
            n_splits=n_partitions, shuffle=True, random_state=123
        )
        for i, (_, client_indexes) in enumerate(
            splitter.split(samples_df.index, samples_df.labels)
        ):
            partition_configs[i] = client_indexes
    return partition_configs


def get_federated_data_loaders(
    labels_file_path: Path,
    data_root_directory: Path,
    training_specs: FederatedTrainingSpecs = FederatedTrainingSpecs(),
    transformations_mode: TransformationsMode = TransformationsMode.AUGMENT,
    augmentations: Optional[Sequence[transforms.Transform]] = None,
    preprocessing_specs: PreprocessingSpecs = PreprocessingSpecs(),
    class_mapping: Optional[Dict[int, int]] = None,
) -> Tuple[Dict[int, DataLoader], List[int]]:
    images, masks, labels = get_images_with_labels(
        data_root_directory, labels_file_path, class_mapping
    )
    samples_df = pd.DataFrame({"images": images, "masks": masks, "labels": labels})
    partitions = get_partitions(
        training_specs.partitioning_mode, training_specs.number_of_clients, samples_df
    )
    data_loaders = {}
    for client_id, indexes in partitions.items():
        data_loaders[client_id] = init_data_loader(
            samples_df.images.iloc[indexes].values,
            samples_df.masks.iloc[indexes].values,
            samples_df.labels.iloc[indexes].values,
            batch_size=training_specs.batch_size,
            transformations_mode=transformations_mode,
            augmentations=augmentations,
            preprocessing_specs=preprocessing_specs,
        )
    return data_loaders, labels


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
            losses_for_clients.append(metrics["training_losses"])
            lr_for_clients.append(metrics["learning_rates"])
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
                np.concatenate([nans_vector, [metrics["loss"]]])
            )
            f_scores_for_clients.append(
                np.concatenate([nans_vector, [metrics["f_score"]]])
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


def load_best_model(
    model: ResNet, trained_model_path: Path, device: torch.device
) -> ResNet:
    saved_parameters = torch.load(trained_model_path, map_location=device)["state_dict"]
    params_dict = zip(model.state_dict().keys(), saved_parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
    return model


def create_strategy(
    fl_strategy_specs: FLStrategySpecs,
    saved_models_folder: Path,
    fit_metrics_aggregation_fn: Callable,
    evaluate_metrics_aggregation_fn: Callable,
) -> SaveModelFedAvg:
    # TODO TODO parameterize it
    strategy = SaveModelFedAvg(
        # fraction_fit=1.0,  # Sample 100% of available clients for training
        # fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
        # min_fit_clients=1,  # Never sample less than 10 clients for training
        # min_evaluate_clients=2,  # Never sample less than 5 clients for evaluation
        # min_available_clients=1,  # Wait until all 10 clients are available
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
    )
    strategy.saved_models_folder = saved_models_folder
    return strategy

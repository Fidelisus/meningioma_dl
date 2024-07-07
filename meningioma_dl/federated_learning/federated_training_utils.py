import logging
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple, Any, Dict, Optional, Sequence, Callable

import numpy as np
import pandas as pd
import torch
from flwr.common import Metrics
from monai import transforms
from torch.utils.data import DataLoader

from meningioma_dl.config import Config
from meningioma_dl.data_loading.data_loader import TransformationsMode, init_data_loader
from meningioma_dl.data_loading.labels_loading import (
    get_images_with_labels,
    get_samples_df,
)
from meningioma_dl.experiments_specs.fl_strategy_specs import FLStrategySpecs
from meningioma_dl.experiments_specs.modelling_specs import ModellingSpecs
from meningioma_dl.experiments_specs.preprocessing_specs import PreprocessingSpecs
from meningioma_dl.experiments_specs.training_specs import FederatedTrainingSpecs
from meningioma_dl.federated_learning.create_federated_data_splits import (
    get_best_split_with_given_ks_stat,
    get_uniform_client_partitions,
    get_non_iid_partitions,
)
from meningioma_dl.federated_learning.server import SaveModelFedAvg, FedProx
from meningioma_dl.models.resnet import ResNet
from meningioma_dl.visualizations.results_visualizations import plot_fl_training_curve


def get_optimizer_and_scheduler(
    parameters_to_fine_tune: List[torch.Tensor],
    modelling_specs: ModellingSpecs,
    learning_rate: float,
) -> Tuple[torch.optim.Optimizer, Any]:
    lr_params = [
        {
            "params": parameters_to_fine_tune,
            "lr": learning_rate,
        }
    ]
    optimizer = torch.optim.Adam(lr_params)
    scheduler = modelling_specs.scheduler_specs.get_scheduler(optimizer)
    return optimizer, scheduler


def get_data_loaders(
    modelling_specs: ModellingSpecs,
    training_specs: FederatedTrainingSpecs,
    manual_seed: int,
) -> Tuple[Dict[int, DataLoader], Dict[int, DataLoader]]:
    train_labels_file_path = Config.train_labels_file_path
    validation_labels_file_path = Config.validation_labels_file_path
    training_data_loader, labels_train = get_federated_data_loaders(
        labels_file_path=train_labels_file_path,
        data_root_directory=Config.data_directory,
        transformations_mode=TransformationsMode.AUGMENT,
        training_specs=training_specs,
        augmentations=modelling_specs.augmentation_specs.transformations_list,
        default_preprocessing_specs=modelling_specs.preprocessing_specs,
        class_mapping=modelling_specs.model_specs.class_mapping,
        manual_seed=manual_seed,
    )
    logging.info(f"Training data loaded from {train_labels_file_path}")
    validation_data_loader, labels_validation = get_federated_data_loaders(
        labels_file_path=validation_labels_file_path,
        data_root_directory=Config.data_directory,
        transformations_mode=TransformationsMode.ONLY_PREPROCESSING,
        training_specs=training_specs,
        default_preprocessing_specs=modelling_specs.preprocessing_specs,
        class_mapping=modelling_specs.model_specs.class_mapping,
        manual_seed=manual_seed,
    )
    logging.info(f"Validation data loaded from {validation_labels_file_path}")
    return training_data_loader, validation_data_loader


def get_federated_data_loaders(
    labels_file_path: Path,
    data_root_directory: Path,
    training_specs: FederatedTrainingSpecs = FederatedTrainingSpecs(),
    transformations_mode: TransformationsMode = TransformationsMode.AUGMENT,
    augmentations: Optional[List[transforms.Transform]] = None,
    default_preprocessing_specs: PreprocessingSpecs = PreprocessingSpecs(),
    class_mapping: Optional[Dict[int, int]] = None,
    manual_seed: int = 123,
) -> Tuple[Dict[int, DataLoader], List[int]]:
    images, masks, labels = get_images_with_labels(
        data_root_directory, labels_file_path, class_mapping
    )
    samples_df = pd.DataFrame({"images": images, "masks": masks, "labels": labels})
    if training_specs.partitioning_mode == "uniform":
        partitions = get_uniform_client_partitions(
            samples_df["labels"], training_specs.number_of_clients, manual_seed
        )
    elif training_specs.partitioning_mode == "ks_stat":
        partitions = get_non_iid_partitions(
            labels_file_path.parent.joinpath(
                f"{labels_file_path.name.split('.')[0]}_"
                f"{training_specs.partitioning_settings['file_name']}.tsv"
            ),
            training_specs.number_of_clients,
        )
        logging.info(f"Partitions: {partitions}")
    else:
        raise ValueError(
            f"Invalid partitioning mode: {training_specs.partitioning_mode}"
        )
    data_loaders = {}
    for client_id, indexes in partitions.items():
        preprocessing = default_preprocessing_specs
        if training_specs.client_specific_preprocessing is not None:
            client_specific_preprocessing = (
                training_specs.client_specific_preprocessing[client_id]
            )
            if client_specific_preprocessing is not None:
                preprocessing = PreprocessingSpecs.get_from_name(
                    client_specific_preprocessing
                )
        data_loaders[client_id] = init_data_loader(
            samples_df.images.iloc[indexes].values,
            samples_df.masks.iloc[indexes].values,
            samples_df.labels.iloc[indexes].values,
            batch_size=training_specs.batch_size,
            transformations_mode=transformations_mode,
            augmentations=augmentations,
            preprocessing_specs=preprocessing,
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
            losses_for_clients.append(np.concatenate([nans_vector, [metrics["loss"]]]))
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
    on_fit_config_fn: Callable,
) -> SaveModelFedAvg:
    if fl_strategy_specs.name == "fed_avg":
        strategy = SaveModelFedAvg(
            fraction_fit=fl_strategy_specs.config.get(
                "fraction_fit", 1.0
            ),  # % of available clients for training
            fraction_eval=fl_strategy_specs.config.get(
                "fraction_eval", 1.0
            ),  # % of available clients for evaluation
            accept_failures=False,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            on_fit_config_fn=on_fit_config_fn,
        )
    elif fl_strategy_specs.name == "fed_prox":
        strategy = FedProx(
            fraction_fit=fl_strategy_specs.config.get(
                "fraction_fit", 1.0
            ),  # % of available clients for training
            fraction_eval=fl_strategy_specs.config.get(
                "fraction_eval", 1.0
            ),  # % of available clients for evaluation
            accept_failures=False,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            on_fit_config_fn=on_fit_config_fn,
            proximal_mu=fl_strategy_specs.config["proximal_mu"],
        )
    else:
        raise KeyError(f"Strategy named {fl_strategy_specs.name} not supported")
    strategy.saved_models_folder = saved_models_folder
    return strategy

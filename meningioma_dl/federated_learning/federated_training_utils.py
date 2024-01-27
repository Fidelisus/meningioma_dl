from pathlib import Path
from typing import List, Tuple, Any, Dict, Optional, Sequence

import pandas as pd
import torch
from monai import transforms
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from meningioma_dl.config import Config
from meningioma_dl.data_loading.data_loader import TransformationsMode, init_data_loader
from meningioma_dl.data_loading.labels_loading import get_images_with_labels

from meningioma_dl.experiments_specs.modelling_specs import ModellingSpecs
from meningioma_dl.experiments_specs.preprocessing_specs import PreprocessingSpecs
from meningioma_dl.experiments_specs.training_specs import FederatedTrainingSpecs
from meningioma_dl.utils import get_loss_function_class_weights


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

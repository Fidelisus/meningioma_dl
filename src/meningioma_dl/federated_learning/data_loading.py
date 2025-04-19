import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import pandas as pd
import torch
from monai import transforms
from torch.utils.data import DataLoader

from meningioma_dl.data_loading.data_loader import (
    TransformationsMode,
    get_client_specific_preprocessing_datasets,
)
from meningioma_dl.data_loading.labels_loading import (
    get_images_with_labels,
)
from meningioma_dl.experiments_specs.modelling_specs import ModellingSpecs
from meningioma_dl.experiments_specs.preprocessing_specs import PreprocessingSpecs
from meningioma_dl.experiments_specs.training_specs import FederatedTrainingSpecs
from meningioma_dl.federated_learning.create_federated_data_splits import (
    get_uniform_client_partitions,
    get_non_iid_partitions,
)


def get_data_loaders(
    modelling_specs: ModellingSpecs,
    training_specs: FederatedTrainingSpecs,
    train_labels_file_path: Path,
    validation_labels_file_path: Path,
    data_directory: Path,
) -> Tuple[Dict[int, DataLoader], Dict[int, DataLoader]]:
    training_data_loader, labels_train = get_federated_data_loaders(
        labels_file_path=train_labels_file_path,
        data_root_directory=data_directory,
        transformations_mode=TransformationsMode.AUGMENT,
        training_specs=training_specs,
        augmentations=modelling_specs.augmentation_specs.transformations_list,
        default_preprocessing_specs=modelling_specs.preprocessing_specs,
        class_mapping=modelling_specs.model_specs.class_mapping,
    )
    logging.info(f"Training data loaded from {train_labels_file_path}")
    validation_data_loader, labels_validation = get_federated_data_loaders(
        labels_file_path=validation_labels_file_path,
        data_root_directory=data_directory,
        transformations_mode=TransformationsMode.ONLY_PREPROCESSING,
        training_specs=training_specs,
        default_preprocessing_specs=modelling_specs.preprocessing_specs,
        class_mapping=modelling_specs.model_specs.class_mapping,
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
    elif training_specs.partitioning_mode == "custom_partitions":
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
    datasets = get_client_specific_preprocessing_datasets(
        samples_df,
        partitions,
        training_specs.client_specific_preprocessing,
        default_preprocessing_specs,
        augmentations,
        transformations_mode,
    )
    data_loaders = {}
    for client_id in partitions:
        data_loaders[client_id] = DataLoader(
            datasets[client_id],
            batch_size=training_specs.batch_size,
            pin_memory=torch.cuda.is_available(),
            shuffle=True,
        )
    return data_loaders, labels

import logging
from pathlib import Path
from typing import Optional

import fire
import torch

from meningioma_dl.config import Config
from meningioma_dl.data_loading.data_loader import get_data_loader, TransformationsMode
from meningioma_dl.model_evaluation.centralized_evaluation import centralized_evaluation
from meningioma_dl.experiments_specs.augmentation_specs import AugmentationSpecs
from meningioma_dl.experiments_specs.model_specs import ModelSpecs
from meningioma_dl.experiments_specs.modelling_specs import (
    ModellingSpecs,
)
from meningioma_dl.experiments_specs.preprocessing_specs import (
    PreprocessingSpecs,
)
from meningioma_dl.experiments_specs.scheduler_specs import SchedulerSpecs
from meningioma_dl.experiments_specs.training_specs import (
    get_training_specs,
    CentralizedTrainingSpecs,
)
from meningioma_dl.models.resnet import create_resnet_model, freeze_resnet_layers
from meningioma_dl.model_training.training_loop import training_loop
from meningioma_dl.jobs.experiments_setup import setup_run, generate_run_id
from meningioma_dl.model_training.loss import get_loss_function


def run_experiment(
    env_file_path: str,
    n_trials: int = 1,
    run_id: str = generate_run_id(),
    device_name: str = "cpu",
    validation_interval: int = 1,
    preprocessing_specs_name: str = "no_resize",
    augmentations_specs_name: str = "basic_01p",
    scheduler_specs_name: str = "05_lr_099_gamma",
    model_specs_name: str = "resnet_10_2_unfreezed",
    training_specs_name: str = "central_1_epochs",
    manual_seed: int = 123,
    cv_fold: Optional[int] = None,
):
    config = setup_run(env_file_path, manual_seed, device_name, cv_fold)

    modelling_spec = ModellingSpecs(
        PreprocessingSpecs.get_from_name(preprocessing_specs_name),
        AugmentationSpecs.get_from_name(augmentations_specs_name),
        SchedulerSpecs.get_from_name(scheduler_specs_name),
        ModelSpecs.get_from_name(model_specs_name),
    )
    training_spec: CentralizedTrainingSpecs = get_training_specs(training_specs_name)
    logging.info(f"run_id: {run_id}")
    logging.info(f"Modelling specs: {modelling_spec}")
    logging.info(f"Augmentations specs name: {augmentations_specs_name}")
    logging.info(f"Training specs: {training_spec}")
    logging.info(f"n_trials: {n_trials}, validation_interval: {validation_interval}")

    visualizations_folder = config.visualizations_directory.joinpath(run_id)
    trained_model_path = train(
        config=config,
        device=config.device,
        validation_interval=validation_interval,
        visualizations_folder=visualizations_folder,
        saved_models_folder=config.saved_models_directory.joinpath(run_id),
        modelling_specs=modelling_spec,
        training_specs=training_spec,
    )
    if trained_model_path is None:
        raise ValueError("No model was created during training, aborting.")

    return centralized_evaluation(
        test_data_path=config.validation_labels_file_path,
        run_id=run_id,
        config=config,
        trained_model_path=trained_model_path,
        device=config.device,
        visualizations_folder=visualizations_folder,
        model_specs=modelling_spec.model_specs,
        preprocessing_specs=modelling_spec.preprocessing_specs,
        training_specs=training_spec,
    )


def train(
    config: Config,
    validation_interval: int = 1,
    device: torch.device = torch.device("cpu"),
    saved_models_folder: Path = Path("."),
    visualizations_folder: Path = Path("."),
    modelling_specs: ModellingSpecs = ModellingSpecs(),
    training_specs: CentralizedTrainingSpecs = CentralizedTrainingSpecs(),
) -> Optional[Path]:
    logging.info(
        f"Starting training with {modelling_specs.model_specs.number_of_classes} classes"
    )
    training_data_loader, labels_train = get_data_loader(
        labels_file_path=config.train_labels_file_path,
        data_root_directory=config.data_directory,
        transformations_mode=TransformationsMode.AUGMENT,
        batch_size=training_specs.batch_size,
        augmentations=modelling_specs.augmentation_specs.transformations_list,
        preprocessing_specs=modelling_specs.preprocessing_specs,
        class_mapping=modelling_specs.model_specs.class_mapping,
        client_specific_preprocessing=training_specs.client_specific_preprocessing,
    )
    logging.info(f"Training data loaded from {config.train_labels_file_path}")
    validation_data_loader, labels_validation = get_data_loader(
        labels_file_path=config.validation_labels_file_path,
        data_root_directory=config.data_directory,
        transformations_mode=TransformationsMode.ONLY_PREPROCESSING,
        batch_size=training_specs.batch_size,
        preprocessing_specs=modelling_specs.preprocessing_specs,
        class_mapping=modelling_specs.model_specs.class_mapping,
        client_specific_preprocessing=training_specs.client_specific_preprocessing,
    )
    logging.info(f"Validation data loaded from {config.validation_labels_file_path}")

    model, pretrained_model_state_dict = create_resnet_model(
        model_depth=modelling_specs.model_specs.model_depth,
        resnet_shortcut_type=modelling_specs.model_specs.resnet_shortcut_type,
        number_of_classes=modelling_specs.model_specs.number_of_classes,
        pretrained_models_directory=config.pretrained_models_directory,
        device=device,
    )
    parameters_names_to_fine_tune, parameters_to_fine_tune = freeze_resnet_layers(
        model=model,
        layers_to_unfreeze=modelling_specs.model_specs.layers_to_unfreeze,
        pretrained_model_state_dict=pretrained_model_state_dict,
    )
    logging.info(f"Parameters to fine tune: {parameters_names_to_fine_tune}")

    lr_params = [
        {
            "params": parameters_to_fine_tune,
            "lr": modelling_specs.scheduler_specs.learning_rate,
        }
    ]
    optimizer = torch.optim.Adam(lr_params)
    scheduler = modelling_specs.scheduler_specs.get_scheduler(optimizer)

    loss_function = get_loss_function(
        labels_train=labels_train,
        labels_validation=labels_validation,
        evaluation_metric_weighting=modelling_specs.model_specs.evaluation_metric_weighting,
    )
    trained_model_path, _ = training_loop(
        training_data_loader=training_data_loader,
        validation_data_loader=validation_data_loader,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_function=loss_function,
        total_epochs=training_specs.epochs,
        validation_interval=validation_interval,
        model_save_folder=saved_models_folder,
        visualizations_folder=visualizations_folder,
        device=device,
        evaluation_metric_weighting=modelling_specs.model_specs.evaluation_metric_weighting,
    )
    return trained_model_path


if __name__ == "__main__":
    try:
        fire.Fire(run_experiment)
    except Exception as e:
        logging.error(f"An unexpected error occured {e}", exc_info=True)
        raise

import logging
from pathlib import Path
from typing import Optional, Tuple

import fire
import torch
from torch import nn

from meningioma_dl.config import Config
from meningioma_dl.data_loading.data_loader import get_data_loader, TransformationsMode
from meningioma_dl.evaluate import evaluate
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
from meningioma_dl.models.resnet import create_resnet_model, freeze_layers
from meningioma_dl.training_utils import training_loop
from meningioma_dl.utils import setup_run, get_loss_function_class_weights


def run_experiment(
    env_file_path: str,
    n_trials: int = 1,
    run_id: Optional[str] = None,
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
    device, run_id = setup_run(env_file_path, run_id, manual_seed, device_name, cv_fold)

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

    visualizations_folder = Config.visualizations_directory.joinpath(run_id)
    _, trained_model_path = train(
        device=device,
        validation_interval=validation_interval,
        visualizations_folder=visualizations_folder,
        saved_models_folder=Config.saved_models_directory.joinpath(run_id),
        modelling_specs=modelling_spec,
        training_specs=training_spec,
    )
    if trained_model_path is None:
        raise ValueError("No model was created during training, aborting.")

    f_score_of_the_best_model = evaluate(
        test_data_path=Config.validation_labels_file_path,
        run_id=run_id,
        trained_model_path=trained_model_path,
        device=device,
        visualizations_folder=visualizations_folder,
        model_specs=modelling_spec.model_specs,
        preprocessing_specs=modelling_spec.preprocessing_specs,
        training_specs=training_spec,
    )
    return f_score_of_the_best_model


def train(
    manual_seed: int = 123,
    validation_interval: int = 1,
    device: torch.device = torch.device("cpu"),
    saved_models_folder: Path = Path("."),
    visualizations_folder: Path = Path("."),
    modelling_specs: ModellingSpecs = ModellingSpecs(),
    training_specs: CentralizedTrainingSpecs = CentralizedTrainingSpecs(),
) -> Tuple[float, Optional[str]]:
    logging.info(
        f"Starting training with {modelling_specs.model_specs.number_of_classes} classes"
    )
    torch.manual_seed(manual_seed)
    train_labels_file_path = Config.train_labels_file_path
    validation_labels_file_path = Config.validation_labels_file_path
    training_data_loader, labels_train = get_data_loader(
        train_labels_file_path,
        Config.data_directory,
        transformations_mode=TransformationsMode.AUGMENT,
        batch_size=training_specs.batch_size,
        augmentations=modelling_specs.augmentation_specs.transformations_list,
        preprocessing_specs=modelling_specs.preprocessing_specs,
        class_mapping=modelling_specs.model_specs.class_mapping,
        client_specific_preprocessing=training_specs.client_specific_preprocessing,
    )
    logging.info(f"Training data loaded from {train_labels_file_path}")
    validation_data_loader, labels_validation = get_data_loader(
        validation_labels_file_path,
        Config.data_directory,
        transformations_mode=TransformationsMode.ONLY_PREPROCESSING,
        batch_size=training_specs.batch_size,
        preprocessing_specs=modelling_specs.preprocessing_specs,
        class_mapping=modelling_specs.model_specs.class_mapping,
        client_specific_preprocessing=training_specs.client_specific_preprocessing,
    )
    logging.info(f"Validation data loaded from {validation_labels_file_path}")

    model, pretrained_model_state_dict = create_resnet_model(
        modelling_specs.model_specs.model_depth,
        modelling_specs.model_specs.resnet_shortcut_type,
        modelling_specs.model_specs.number_of_classes,
        Config.pretrained_models_directory,
        device,
    )
    parameters_names_to_fine_tune, parameters_to_fine_tune = freeze_layers(
        model,
        modelling_specs.model_specs.number_of_layers_to_unfreeze,
        pretrained_model_state_dict,
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

    loss_function_weighting = (
        torch.tensor(
            get_loss_function_class_weights(labels_train + labels_validation)
        ).to(torch.float64)
        if modelling_specs.model_specs.evaluation_metric_weighting == "weighted"
        else None
    )
    loss_function = nn.CrossEntropyLoss(
        weight=loss_function_weighting,
    )
    loss_function = loss_function.to(device)

    logging.info("Model initialized succesfully")

    best_f_score, trained_model_path, _ = training_loop(
        training_data_loader,
        validation_data_loader,
        model,
        optimizer,
        scheduler,
        loss_function,
        total_epochs=training_specs.epochs,
        validation_interval=validation_interval,
        model_save_folder=saved_models_folder,
        visualizations_folder=visualizations_folder,
        device=device,
        evaluation_metric_weighting=modelling_specs.model_specs.evaluation_metric_weighting,
    )
    return best_f_score, trained_model_path


if __name__ == "__main__":
    try:
        fire.Fire(run_experiment)
    except Exception as e:
        logging.error(f"An unexpected error occured {e}", exc_info=True)
        raise

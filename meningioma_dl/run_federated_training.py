import numpy as np
import torch

import copy
import logging
from collections import OrderedDict
from functools import partial
from logging import INFO
from pathlib import Path
from typing import Tuple, Optional, Callable, List, Dict

import fire
import flwr as fl
from flwr.common import Metrics
from flwr.common.logger import log
from flwr.server import History

from meningioma_dl.config import Config
from meningioma_dl.data_loading.data_loader import TransformationsMode, get_data_loader
from meningioma_dl.evaluate import evaluate_model
from meningioma_dl.experiments_specs.augmentation_specs import AugmentationSpecs
from meningioma_dl.experiments_specs.fl_strategy_specs import FLStrategySpecs
from meningioma_dl.experiments_specs.model_specs import ModelSpecs
from meningioma_dl.experiments_specs.modelling_specs import ModellingSpecs
from meningioma_dl.experiments_specs.preprocessing_specs import PreprocessingSpecs
from meningioma_dl.experiments_specs.scheduler_specs import SchedulerSpecs
from meningioma_dl.experiments_specs.training_specs import (
    FederatedTrainingSpecs,
    get_training_specs,
)
from meningioma_dl.federated_learning.clients import ClassicalFLClient
from meningioma_dl.federated_learning.federated_training_utils import (
    get_data_loaders,
    visualize_federated_learning_metrics,
    create_strategy,
    load_best_model,
)
from meningioma_dl.models.resnet import create_resnet_model, ResNet, freeze_layers
from meningioma_dl.training_utils import training_loop
from meningioma_dl.utils import (
    setup_run,
    setup_logging,
    generate_run_id,
    setup_flower_logger,
)


class FederatedTraining:
    device: Optional[torch.device]
    model: Optional[ResNet]
    pretrained_model_state_dict: Dict
    training_function: Optional[Callable]
    evaluation_function: Optional[Callable]

    def __init__(
        self,
        modelling_specs: ModellingSpecs = ModellingSpecs(),
        training_specs: FederatedTrainingSpecs = FederatedTrainingSpecs(),
        fl_strategy_specs: FLStrategySpecs = FLStrategySpecs(),
        saved_models_folder: Path = Path("."),
        visualizations_folder: Path = Path("."),
    ):
        self.modelling_specs: ModellingSpecs = modelling_specs
        self.training_specs: FederatedTrainingSpecs = training_specs
        self.fl_strategy_specs: FLStrategySpecs = fl_strategy_specs
        self.visualizations_folder: Path = visualizations_folder
        self.saved_models_folder: Path = saved_models_folder
        self._init_instance_variables()

    def _init_instance_variables(self):
        self.training_metrics = []
        self.validation_metrics = []
        self.last_lr: Optional[float] = None

    def _save_fit_metrics(
        self, metrics_from_clients: List[Tuple[int, Metrics]]
    ) -> Metrics:
        self.training_metrics.append(metrics_from_clients)
        if not self.training_specs.reset_learning_rate_every_round:
            self.last_lr = metrics_from_clients[0][1]["learning_rates"][-1]
        return {}

    def _visualize_federated_learning_metrics(
        self, validation_metrics_from_clients: List[Tuple[int, Metrics]]
    ):
        self.validation_metrics.append(validation_metrics_from_clients)
        visualize_federated_learning_metrics(
            self.training_metrics,
            self.validation_metrics,
            self.visualizations_folder,
            self.training_specs.epochs_per_round,
        )
        n_samples_per_client = []
        f_score_per_client = []
        for n_samples, metrics in validation_metrics_from_clients:
            n_samples_per_client.append(n_samples)
            f_score_per_client.append(metrics["f_score"])
        weighted_f_score = np.average(f_score_per_client, weights=n_samples_per_client)
        return {"f_score": weighted_f_score}

    def _set_clients_train_and_eval_functions(self, run_id):
        clients_logging_function = partial(log, INFO)
        self.training_function = partial(
            training_loop,
            validation_data_loader=None,
            total_epochs=self.training_specs.epochs_per_round,
            validation_interval=None,
            model_save_folder=None,
            device=self.device,
            evaluation_metric_weighting=self.modelling_specs.model_specs.evaluation_metric_weighting,
            logger=clients_logging_function,
            visualizations_folder=self.visualizations_folder,
            save_images=False,
        )
        self.evaluation_function = partial(
            evaluate_model,
            model_specs=self.modelling_specs.model_specs,
            training_specs=self.training_specs,
            device=self.device,
            run_id=run_id,
            logger=clients_logging_function,
            visualizations_folder=self.visualizations_folder,
            save_images=False,
        )

    def on_fit_config_fn(self, _) -> Dict[str, float]:
        return {"last_lr": self.last_lr}

    def client_fn(self, cid: str) -> ClassicalFLClient:
        model = copy.deepcopy(self.model).to(self.device)

        training_data_loader = self.training_data_loaders[int(cid)]
        validation_data_loader = self.validation_data_loaders[int(cid)]

        return ClassicalFLClient(
            cid=int(cid),
            model=model,
            training_data_loader=training_data_loader,
            validation_data_loader=validation_data_loader,
            modelling_specs=self.modelling_specs,
            pretrained_model_state_dict=self.pretrained_model_state_dict,
            device=self.device,
            training_function=self.training_function,
            evaluation_function=self.evaluation_function,
            visualizations_folder=self.visualizations_folder.joinpath(f"client_{cid}"),
        )

    def _evaluate_best_model(self, trained_model_path: Path, run_id: str):
        self.model = load_best_model(self.model, trained_model_path, self.device)
        validation_data_loader, _ = get_data_loader(
            Config.validation_labels_file_path,
            Config.data_directory,
            transformations_mode=TransformationsMode.ONLY_PREPROCESSING,
            batch_size=self.training_specs.batch_size,
            preprocessing_specs=self.modelling_specs.preprocessing_specs,
            class_mapping=self.modelling_specs.model_specs.class_mapping,
        )
        evaluate_model(
            model=self.model,
            data_loader=validation_data_loader,
            model_specs=self.modelling_specs.model_specs,
            training_specs=self.training_specs,
            device=self.device,
            run_id=run_id,
            logger=logging.info,
            visualizations_folder=self.visualizations_folder,
        )

    def run_federated_training(
        self,
        run_id: Optional[str] = None,
        manual_seed: int = 123,
        device: torch.device = torch.device("cpu"),
    ) -> History:
        self.device = device
        self._init_instance_variables()
        setup_flower_logger()

        torch.manual_seed(manual_seed)
        logging.info("Starting federated training")
        (
            self.training_data_loaders,
            self.validation_data_loaders,
        ) = get_data_loaders(self.modelling_specs, self.training_specs, manual_seed)
        self.model, self.pretrained_model_state_dict = create_resnet_model(
            self.modelling_specs.model_specs.model_depth,
            self.modelling_specs.model_specs.resnet_shortcut_type,
            self.modelling_specs.model_specs.number_of_classes,
            Config.pretrained_models_directory,
            self.device,
        )
        self._set_clients_train_and_eval_functions(run_id)
        self.last_lr = self.modelling_specs.scheduler_specs.learning_rate

        logging.info("Global model initialized succesfully")

        strategy = create_strategy(
            self.fl_strategy_specs,
            fit_metrics_aggregation_fn=self._save_fit_metrics,
            evaluate_metrics_aggregation_fn=self._visualize_federated_learning_metrics,
            saved_models_folder=self.saved_models_folder,
            on_fit_config_fn=self.on_fit_config_fn,
        )
        client_resources = self._get_client_resources()

        logging.info("Starting simulation")
        training_history = fl.simulation.start_simulation(
            client_fn=self.client_fn,
            num_clients=self.training_specs.number_of_clients,
            num_rounds=self.training_specs.global_epochs,
            strategy=strategy,
            client_resources=client_resources,
            # ray_init_args={
            #     "ignore_reinit_error": True,
            #     # "local_mode": True,
            #     "include_dashboard": False,
            #     # "address": "127.0.0.1:10001",
            #     "num_cpus": 8,
            #     # "num_gpus":1
            # }
        )

        self._evaluate_best_model(strategy.trained_model_path, run_id)

        return training_history

    def _get_client_resources(self):
        client_resources = {"num_gpus": 0, "num_cpus": 4}
        if self.device.type == "cuda":
            client_resources = {"num_gpus": 1, "num_cpus": 2}
        return client_resources


def main(
    env_file_path: Optional[str] = None,
    run_id: Optional[str] = None,
    device_name: str = "cpu",
    preprocessing_specs_name: str = "no_resize",
    augmentations_specs_name: str = "basic_01p",
    scheduler_specs_name: str = "05_lr_099_gamma",
    model_specs_name: str = "resnet_10_2_unfreezed",
    training_specs_name: str = "federated_local_run",
    fl_strategy_specs_name: str = "fed_avg_default",
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
    training_spec: FederatedTrainingSpecs = get_training_specs(training_specs_name)
    fl_strategy_specs: FLStrategySpecs = FLStrategySpecs.get_from_name(
        fl_strategy_specs_name
    )
    logging.info(f"run_id: {run_id}")
    logging.info(f"Modelling specs: {modelling_spec}")
    logging.info(f"Augmentations specs name: {augmentations_specs_name}")
    logging.info(f"Training specs: {training_spec}")
    logging.info(f"FL strategy specs: {fl_strategy_specs}")

    trainer = FederatedTraining(
        modelling_spec,
        training_spec,
        fl_strategy_specs,
        visualizations_folder=Config.visualizations_directory.joinpath(run_id),
        saved_models_folder=Config.saved_models_directory.joinpath(run_id),
    )
    trainer.run_federated_training(
        run_id=run_id,
        device=device,
        manual_seed=manual_seed,
    )
    logging.info(f"Training for {run_id} finished successfully.")


if __name__ == "__main__":
    try:
        fire.Fire(main)
    except Exception as e:
        logging.error(f"An unexpected error occured {e}", exc_info=True)
        raise

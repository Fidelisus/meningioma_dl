import copy
import json
import logging
from collections import defaultdict
from functools import partial
from logging import INFO
from pathlib import Path
from typing import Tuple, Optional, Callable, List, Dict, Any, Literal

import fire
import flwr as fl
import numpy as np
import torch
from flwr.client import Client
from flwr.common import Metrics, Context
from flwr.common.logger import log
from flwr.server import History, ServerConfig
from torch.utils.data import DataLoader

from meningioma_dl.config import Config
from meningioma_dl.data_loading.data_loader import TransformationsMode, get_data_loader
from meningioma_dl.evaluate_ensemble import evaluate_ensemble
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
from meningioma_dl.federated_learning.ensemble_fl import (
    get_local_ensemble_weights,
    get_global_ensemble_weights,
    ensemble_weights_to_numpy,
)
from meningioma_dl.federated_learning.data_loading import (
    get_data_loaders,
)
from meningioma_dl.federated_learning.strategy import create_strategy
from meningioma_dl.federated_learning.visualizations import (
    visualize_federated_learning_metrics,
)
from meningioma_dl.model_evaluation.centralized_evaluation import evaluate
from meningioma_dl.model_evaluation.metrics import calculate_basic_metrics
from meningioma_dl.model_training.predictions import get_model_predictions
from meningioma_dl.model_training.training_loop import (
    training_loop,
)
from meningioma_dl.models.resnet import (
    create_resnet_model,
    ResNet,
    load_model_from_file,
)
from meningioma_dl.data_loading.experiments_setup import (
    setup_run,
    setup_flower_logger,
    generate_run_id,
)
from meningioma_dl.visualizations.results_visualizations import (
    create_evaluation_report,
    deserialize_value,
)


class FederatedTraining:
    device: Optional[torch.device]
    model: Optional[ResNet]
    pretrained_model_state_dict: Dict
    training_function: Optional[Callable]
    evaluation_function: Optional[Callable]

    def __init__(
        self,
        config: Config,
        modelling_specs: ModellingSpecs = ModellingSpecs(),
        training_specs: FederatedTrainingSpecs = FederatedTrainingSpecs(),
        fl_strategy_specs: FLStrategySpecs = FLStrategySpecs(),
    ):
        self.config = config
        self.modelling_specs: ModellingSpecs = modelling_specs
        self.training_specs: FederatedTrainingSpecs = training_specs
        self.fl_strategy_specs: FLStrategySpecs = fl_strategy_specs
        self._reset_instance_variables()

    def _reset_instance_variables(self):
        self.training_metrics = []
        self.validation_metrics = []
        self.last_lr: Optional[float] = None

    def _save_fit_metrics(
        self, metrics_from_clients: List[Tuple[int, Metrics]]
    ) -> Metrics:
        self.training_metrics.append(metrics_from_clients)
        if not self.training_specs.reset_learning_rate_every_round:
            self.last_lr = metrics_from_clients[0][1]["last_lr"]
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
            f_score_per_client.append(deserialize_value(metrics, "f_score"))
        weighted_f_score = np.average(f_score_per_client, weights=n_samples_per_client)
        return {"f_score": weighted_f_score}

    def _get_clients_train_and_eval_functions(
        self, run_id: str
    ) -> tuple[Callable, Callable]:
        clients_logging_function = partial(log, INFO)
        validation_interval = (
            1 if self.fl_strategy_specs.name == "fed_ensemble" else None
        )
        training_function = partial(
            training_loop,
            total_epochs=self.training_specs.epochs_per_round,
            validation_interval=validation_interval,
            model_save_folder=None,
            device=self.config.device,
            evaluation_metric_weighting=self.modelling_specs.model_specs.evaluation_metric_weighting,
            logger=clients_logging_function,
            visualizations_folder=self.visualizations_folder,
            save_images=False,
        )
        evaluation_function = partial(
            evaluate,
            model_specs=self.modelling_specs.model_specs,
            training_specs=self.training_specs,
            device=self.config.device,
            run_id=run_id,
            logger=clients_logging_function,
            visualizations_folder=self.visualizations_folder,
            save_images=True,
        )
        return training_function, evaluation_function

    def on_fit_config_fn(self, _) -> Dict[str, float]:
        return {"last_lr": self.last_lr}

    def client_fn(self, context: Context) -> Client:
        model = copy.deepcopy(self.model).to(self.config.device)
        data_loader_id = int(context.node_config["partition-id"])
        training_data_loader = self.training_data_loaders[data_loader_id]
        validation_data_loader = self.validation_data_loaders[data_loader_id]

        return ClassicalFLClient(
            cid=data_loader_id,
            model=model,
            training_data_loader=training_data_loader,
            validation_data_loader=validation_data_loader,
            modelling_specs=self.modelling_specs,
            pretrained_model_state_dict=self.pretrained_model_state_dict,
            device=self.config.device,
            training_function=self.training_function,
            evaluation_function=self.evaluation_function,
            visualizations_folder=self.visualizations_folder.joinpath(
                f"client_{data_loader_id}"
            ),
        ).to_client()

    def _evaluate_model(
        self, validation_data_loader: DataLoader, trained_model_path: Path, run_id: str
    ):
        self.model = load_model_from_file(
            trained_model_path, self.modelling_specs.model_specs, self.config.device
        )
        evaluate(
            model=self.model,
            data_loader=validation_data_loader,
            model_specs=self.modelling_specs.model_specs,
            training_specs=self.training_specs,
            device=self.config.device,
            run_id=run_id,
            logger=logging.info,
            visualizations_folder=self.visualizations_folder,
        )

    def _save_as_json(self, file_name: str, object_to_save: Any) -> None:
        with open(
            self.visualizations_folder.joinpath(file_name),
            "w",
        ) as f:
            json.dump(object_to_save, f)

    # TODO TODO move to ensemble_fl
    def _get_local_models_vs_clients_f_scores(
        self, saved_models_folder: Path, device: torch.device
    ) -> Tuple[Dict[int, ResNet], Dict[int, Dict[int, float]]]:
        clients_models: Dict[int, ResNet] = {}
        local_models_vs_clients_f_scores: Dict[int, Dict[int, float]] = defaultdict(
            dict
        )
        for client_id in range(self.training_specs.number_of_clients):
            client_model = load_model_from_file(
                saved_models_folder.joinpath(f"best_model{client_id}.pth.tar"),
                self.modelling_specs.model_specs,
                torch.device("cpu"),
            )
            clients_models[client_id] = client_model
            for client_id_to_validate_on in range(
                self.training_specs.number_of_clients
            ):
                data_loader = self.validation_data_loaders[client_id_to_validate_on]
                client_model.eval()
                client_model = client_model.to(device)
                with torch.no_grad():
                    labels, predictions, images_paths = get_model_predictions(
                        data_loader, client_model, device
                    )
                predictions_flat = predictions.cpu().argmax(dim=1)
                f_score = calculate_basic_metrics(
                    labels.cpu(),
                    predictions_flat,
                    self.modelling_specs.model_specs.evaluation_metric_weighting,
                    logging.info,
                )
                local_models_vs_clients_f_scores[client_id][
                    client_id_to_validate_on
                ] = f_score
        return clients_models, local_models_vs_clients_f_scores

    def _get_client_resources(self):
        client_resources = {"num_gpus": 0, "num_cpus": 4}
        if self.config.device.type == "cuda":
            client_resources = {"num_gpus": 1, "num_cpus": 2}
        return client_resources

    def _setup_run(self, run_id: str, manual_seed: int) -> None:
        self.visualizations_folder = self.config.visualizations_directory.joinpath(
            run_id
        )
        self.saved_models_folder = self.config.saved_models_directory.joinpath(run_id)
        self._reset_instance_variables()
        setup_flower_logger()
        torch.manual_seed(manual_seed)

    def _evaluate_fed_ensemble_model(
        self, validation_data_loader: DataLoader, saved_models_folder: Path, run_id: str
    ) -> None:
        (
            clients_models,
            local_models_vs_clients_f_scores,
        ) = self._get_local_models_vs_clients_f_scores(
            saved_models_folder, self.config.device
        )
        self._save_as_json(
            "local_models_vs_clients_f_scores.json",
            local_models_vs_clients_f_scores,
        )
        global_ensemble_weights = get_global_ensemble_weights(
            local_models_vs_clients_f_scores
        )
        local_ensemble_weights = get_local_ensemble_weights(
            local_models_vs_clients_f_scores
        )
        self._save_as_json("global_ensemble_weights.json", global_ensemble_weights)
        for client_id in local_ensemble_weights:
            self._save_as_json(
                f"local_ensemble_weights_{client_id}.json",
                local_ensemble_weights[client_id],
            )
        logging.info("Evaluating global ensemble: ")
        _, validation_metrics = evaluate_ensemble(
            data_loader=validation_data_loader,
            models=list(clients_models.values()),
            model_specs=self.modelling_specs.model_specs,
            device=self.config.device,
            ensemble_models_weights=ensemble_weights_to_numpy(global_ensemble_weights),
        )
        create_evaluation_report(
            true=validation_metrics.true,
            predictions=validation_metrics.predictions,
            visualizations_folder=self.visualizations_folder,
            run_id=run_id,
            model_specs=self.modelling_specs.model_specs,
            training_specs=self.training_specs,
            n_classes=self.modelling_specs.model_specs.number_of_classes,
        )

    def _evaluate_best_model(self, run_id, strategy):
        final_model_validation_data_loader, _ = get_data_loader(
            labels_file_path=self.config.validation_labels_file_path,
            data_root_directory=self.config.data_directory,
            transformations_mode=TransformationsMode.ONLY_PREPROCESSING,
            batch_size=self.training_specs.batch_size,
            preprocessing_specs=self.modelling_specs.preprocessing_specs,
            class_mapping=self.modelling_specs.model_specs.class_mapping,
        )
        if self.fl_strategy_specs.name == "fed_ensemble":
            self._evaluate_fed_ensemble_model(
                final_model_validation_data_loader, strategy.saved_models_folder, run_id
            )
        else:
            self._evaluate_model(
                final_model_validation_data_loader, strategy.trained_model_path, run_id
            )

    def _load_user_pretrained_model(self, pretrained_model_run_id: str) -> ResNet:
        pretrained_model_file = self.saved_models_folder.parent.joinpath(
            pretrained_model_run_id, "best_model.pth.tar"
        )
        model = load_model_from_file(
            pretrained_model_file,
            self.modelling_specs.model_specs,
            self.config.device,
        )
        logging.info(f"Pretrained model loaded from {pretrained_model_file}")
        return model

    def run_federated_training(self, run_id: str, manual_seed: int = 123) -> None:
        self._setup_run(run_id, manual_seed)

        logging.info("Starting federated training")
        (
            self.training_data_loaders,
            self.validation_data_loaders,
        ) = get_data_loaders(
            modelling_specs=self.modelling_specs,
            training_specs=self.training_specs,
            train_labels_file_path=self.config.train_labels_file_path,
            validation_labels_file_path=self.config.validation_labels_file_path,
            data_directory=self.config.data_directory,
        )

        self.model, self.pretrained_model_state_dict = create_resnet_model(
            model_depth=self.modelling_specs.model_specs.model_depth,
            resnet_shortcut_type=self.modelling_specs.model_specs.resnet_shortcut_type,
            number_of_classes=self.modelling_specs.model_specs.number_of_classes,
            pretrained_models_directory=self.config.pretrained_models_directory,
            device=self.config.device,
        )
        if pretrained_model_run_id := self.fl_strategy_specs.config.get(
            "pretrained_model_run_id"
        ):
            self.model = self._load_user_pretrained_model(pretrained_model_run_id)

        self.training_function, self.evaluation_function = (
            self._get_clients_train_and_eval_functions(run_id)
        )
        self.last_lr = self.modelling_specs.scheduler_specs.learning_rate

        strategy = create_strategy(
            fl_strategy_specs=self.fl_strategy_specs,
            fit_metrics_aggregation_fn=self._save_fit_metrics,
            evaluate_metrics_aggregation_fn=self._visualize_federated_learning_metrics,
            saved_models_folder=self.saved_models_folder,
            on_fit_config_fn=self.on_fit_config_fn,
        )
        client_resources = self._get_client_resources()

        logging.info("Starting simulation")
        fl.simulation.start_simulation(
            client_fn=self.client_fn,
            num_clients=self.training_specs.number_of_clients,
            config=ServerConfig(num_rounds=self.training_specs.global_epochs),
            strategy=strategy,
            client_resources=client_resources,
            ray_init_args={
                "ignore_reinit_error": True,
                "include_dashboard": False,
                "num_cpus": 4,
            },
        )
        self._evaluate_best_model(run_id, strategy)


def run_experiment(
    env_file_path: str,
    run_id: str = generate_run_id(),
    device_name: Literal["cpu", "cuda"] = "cpu",
    preprocessing_specs_name: str = "no_resize",
    augmentations_specs_name: str = "basic_01p",
    scheduler_specs_name: str = "05_lr_099_gamma",
    model_specs_name: str = "resnet_10_2_unfreezed",
    training_specs_name: str = "federated_local_run",
    fl_strategy_specs_name: str = "fed_avg_default",
    manual_seed: int = 123,
    cv_fold: int = 0,
):
    config = setup_run(
        env_file_path=env_file_path,
        manual_seed=manual_seed,
        device_name=device_name,
        cv_fold=cv_fold,
    )
    modelling_specs = ModellingSpecs(
        PreprocessingSpecs.get_from_name(preprocessing_specs_name),
        AugmentationSpecs.get_from_name(augmentations_specs_name),
        SchedulerSpecs.get_from_name(scheduler_specs_name),
        ModelSpecs.get_from_name(model_specs_name),
    )
    training_specs: FederatedTrainingSpecs = get_training_specs(training_specs_name)
    fl_strategy_specs: FLStrategySpecs = FLStrategySpecs.get_from_name(
        fl_strategy_specs_name
    )
    logging.info(f"run_id: {run_id}")
    logging.info(f"Modelling specs: {modelling_specs}")
    logging.info(f"Augmentations specs name: {augmentations_specs_name}")
    logging.info(f"Training specs: {training_specs}")
    logging.info(f"FL strategy specs: {fl_strategy_specs}")

    trainer = FederatedTraining(
        modelling_specs=modelling_specs,
        training_specs=training_specs,
        fl_strategy_specs=fl_strategy_specs,
        config=config,
    )
    trainer.run_federated_training(run_id=run_id, manual_seed=manual_seed)
    logging.info(f"Training for {run_id} finished successfully.")


if __name__ == "__main__":
    try:
        fire.Fire(run_experiment)
    except Exception as e:
        logging.error(f"An unexpected error occured {e}", exc_info=True)
        raise

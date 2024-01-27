import logging
from functools import partial
from logging import INFO
from pathlib import Path
from typing import Tuple, Optional, Callable, List, Any

import fire
import flwr as fl
import torch
from flwr.common.logger import log
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from meningioma_dl.config import Config
from meningioma_dl.evaluate import evaluate_model
from meningioma_dl.experiments_specs.modelling_specs import ModellingSpecs
from meningioma_dl.experiments_specs.training_specs import (
    FederatedTrainingSpecs,
)
from meningioma_dl.federated_learning.clients import ClassicalFLClient
from meningioma_dl.federated_learning.federated_training_utils import get_data_loaders
from meningioma_dl.models.resnet import create_resnet_model, ResNet
from meningioma_dl.training_utils import training_loop
from meningioma_dl.utils import (
    setup_run,
)


class FederatedTraining:
    def __init__(
        self,
        base_visualizations_folder: Path = Path("."),
        modelling_specs: ModellingSpecs = ModellingSpecs(),
        training_specs: FederatedTrainingSpecs = FederatedTrainingSpecs(),
        saved_models_folder: Path = Path("."),
    ):
        self.modelling_specs: ModellingSpecs = modelling_specs
        self.training_specs: FederatedTrainingSpecs = training_specs
        self.base_visualizations_folder: Path = base_visualizations_folder

        self.device: Optional[torch.device] = None
        self.model: Optional[ResNet] = None
        self.loss_function_weighting: Optional[torch.Tensor] = None
        self.parameters_to_fine_tune: Optional[List[torch.Tensor]] = None

        self.training_function: Optional[
            Callable[
                [DataLoader, ResNet, Optimizer, Any, Any], Tuple[float, Optional[Path]]
            ]
        ] = None
        self.evaluation_function: Optional[
            Callable[[DataLoader, ResNet, Path], float]
        ] = None

    def run_federated_training(
        self,
        env_file_path: Optional[str] = None,
        run_id: Optional[str] = None,
        manual_seed: int = Config.random_seed,
        validation_interval: int = 1,
        device_name: str = "cpu",
    ):
        self.device = setup_run(env_file_path, run_id, manual_seed, device_name)

        logging.info(
            "Starting federated training with "
            f"{self.modelling_specs.model_specs.number_of_classes} classes"
        )
        (
            self.training_data_loaders,
            self.validation_data_loaders,
            self.loss_function_weighting,
        ) = get_data_loaders(self.modelling_specs, self.training_specs)

        self.model, self.parameters_to_fine_tune = create_resnet_model(
            self.modelling_specs.model_specs.model_depth,
            self.modelling_specs.model_specs.resnet_shortcut_type,
            self.modelling_specs.model_specs.number_of_classes,
            Config.pretrained_models_directory,
            self.device,
            self.modelling_specs.model_specs.number_of_layers_to_unfreeze,
        )

        logging.info("Global model initialized succesfully")

        logging_function = partial(log, INFO)
        self.training_function = partial(
            training_loop,
            validation_data_loader=None,
            total_epochs=self.training_specs.epochs_per_client,
            validation_interval=None,
            model_save_folder=None,
            device=self.device,
            evaluation_metric_weighting=self.modelling_specs.model_specs.evaluation_metric_weighting,
            logger=logging_function,
        )
        self.evaluation_function = partial(
            evaluate_model,
            modelling_specs=self.modelling_specs,
            training_specs=self.training_specs,
            device=self.device,
            run_id=run_id,
            logger=logging_function,
        )

        # TODO TODO parameterize it
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=1.0,  # Sample 100% of available clients for training
            fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
            min_fit_clients=2,  # Never sample less than 10 clients for training
            min_evaluate_clients=2,  # Never sample less than 5 clients for evaluation
            min_available_clients=2,  # Wait until all 10 clients are available
        )

        client_resources = {"num_gpus": 0, "num_cpus": 4}
        if self.device.type == "cuda":
            client_resources = {"num_gpus": 1, "num_cpus": 4}

        training_history = fl.simulation.start_simulation(
            client_fn=self.client_fn,
            num_clients=self.training_specs.number_of_clients,
            config=fl.server.ServerConfig(num_rounds=self.training_specs.global_epochs),
            strategy=strategy,
            client_resources=client_resources,
        )

        return training_history

    def client_fn(self, cid: str) -> ClassicalFLClient:
        model = self.model.to(self.device)

        training_data_loader = self.training_data_loaders[int(cid)]
        validation_data_loader = self.validation_data_loaders[int(cid)]

        return ClassicalFLClient(
            cid=int(cid),
            model=model,
            training_data_loader=training_data_loader,
            validation_data_loader=validation_data_loader,
            modelling_specs=self.modelling_specs,
            parameters_to_fine_tune=self.parameters_to_fine_tune,
            loss_function_weighting=self.loss_function_weighting,
            device=self.device,
            training_function=self.training_function,
            evaluation_function=self.evaluation_function,
            visualizations_folder=self.base_visualizations_folder.joinpath(
                f"client_{cid}"
            ),
        )


def main(env_file_path: Optional[str] = None):
    trainer = FederatedTraining()
    trainer.run_federated_training(env_file_path)


if __name__ == "__main__":
    try:
        fire.Fire(main)
    except Exception as e:
        logging.error(f"An unexpected error occured {e}", exc_info=True)
        raise

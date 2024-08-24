from dataclasses import asdict
from pathlib import Path
from typing import Tuple, Dict, Callable, Optional, Any

import logging
import flwr as fl
import torch
from flwr.common import Scalar, Config
from torch import nn
from torch.utils.data import DataLoader

from meningioma_dl.experiments_specs.modelling_specs import ModellingSpecs
from meningioma_dl.models.resnet import ResNet, freeze_layers, set_model_parameters
from meningioma_dl.federated_learning.federated_training_utils import (
    get_optimizer_and_scheduler,
)


def get_model_parameters(model: ResNet):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


class ClassicalFLClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid: int,
        model: ResNet,
        training_data_loader: DataLoader,
        validation_data_loader: DataLoader,
        modelling_specs: Optional[ModellingSpecs],
        pretrained_model_state_dict: Dict,
        device: torch.device,
        visualizations_folder: Path,
        training_function: Callable,
        evaluation_function: Callable,
    ):
        self.cid = cid
        self.model = model
        self.training_data_loader = training_data_loader
        self.validation_data_loader = validation_data_loader
        self.modelling_specs = modelling_specs
        self.pretrained_model_state_dict = pretrained_model_state_dict
        self.device = device
        self.visualizations_folder = visualizations_folder

        self.training_function = training_function
        self.evaluation_function = evaluation_function

    def get_parameters(self, config: Config):
        logging.info(f"[Client {self.cid}] get_parameters")
        try:
            parameters = get_model_parameters(self.model)
        except Exception as e:
            logging.error(
                f"Getting model parameters at client {self.cid} failed with error {e}",
                exc_info=True,
            )
            raise e
        return parameters

    def fit(
        self, parameters, config: Dict[str, Scalar]
    ) -> Tuple[Any, int, Dict[str, Scalar]]:
        logging.info(f"[Client {self.cid}] fit, config: {config}")
        try:
            set_model_parameters(self.model, parameters)
            _, parameters_to_fine_tune = freeze_layers(
                self.model,
                self.modelling_specs.model_specs.number_of_layers_to_unfreeze,
                self.pretrained_model_state_dict,
            )
            optimizer, scheduler = get_optimizer_and_scheduler(
                parameters_to_fine_tune,
                self.modelling_specs,
                config["last_lr"],
            )
            loss_function = nn.CrossEntropyLoss().to(self.device)
            _, _, training_metrics = self.training_function(
                training_data_loader=self.training_data_loader,
                model=self.model,
                optimizer=optimizer,
                scheduler=scheduler,
                loss_function=loss_function,
                visualizations_folder=self.visualizations_folder,
                proximal_mu=config.get("proximal_mu", None),
            )
        except Exception as e:
            logging.error(
                f"Fitting at client {self.cid} failed with error {e}", exc_info=True
            )
            raise e
        return (
            get_model_parameters(self.model),
            len(self.training_data_loader),
            asdict(training_metrics),
        )

    def evaluate(
        self, parameters, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        logging.info(f"[Client {self.cid}] evaluate, config: {config}")
        set_model_parameters(self.model, parameters)

        try:
            f1_score, validation_metrics = self.evaluation_function(
                data_loader=self.validation_data_loader,
                model=self.model,
                loss_function=nn.CrossEntropyLoss().to(self.device),
                visualizations_folder=self.visualizations_folder,
            )
        except Exception as e:
            logging.error(
                f"Validation at client {self.cid} failed with error {e}", exc_info=True
            )
            raise e
        return (
            float(f1_score),
            len(self.validation_data_loader),
            asdict(validation_metrics),
        )

from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
from typing import Tuple, List, Dict, Callable, Optional, Any

import flwr as fl
import torch
from flwr.common import Scalar
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from meningioma_dl.experiments_specs.modelling_specs import ModellingSpecs
from meningioma_dl.models.resnet import ResNet
from meningioma_dl.federated_learning.federated_training_utils import (
    get_optimizer_and_scheduler,
)
from meningioma_dl.visualizations.results_visualizations import TrainingMetrics


def get_model_parameters(model: ResNet):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_model_parameters(model: ResNet, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


class ClassicalFLClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid: int,
        model: ResNet,
        training_data_loader: DataLoader,
        validation_data_loader: DataLoader,
        modelling_specs: Optional[ModellingSpecs],
        parameters_to_fine_tune: Optional[List[torch.Tensor]],
        loss_function_weighting: Optional[torch.Tensor],
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
        self.parameters_to_fine_tune = parameters_to_fine_tune
        self.loss_function_weighting = loss_function_weighting
        self.device = device
        self.visualizations_folder = visualizations_folder

        self.training_function = training_function
        self.evaluation_function = evaluation_function

    def get_parameters(self):
        print(f"[Client {self.cid}] get_parameters")
        return get_model_parameters(self.model)

    def fit(
        self, parameters, config: Dict[str, Scalar]
    ) -> Tuple[Any, int, Dict[str, Scalar]]:
        print(f"[Client {self.cid}] fit, config: {config}")
        set_model_parameters(self.model, parameters)

        optimizer, scheduler = get_optimizer_and_scheduler(
            self.parameters_to_fine_tune,
            self.modelling_specs,
            config["last_lr"],
        )
        loss_function = nn.CrossEntropyLoss(
            weight=self.loss_function_weighting,
        ).to(self.device)

        _, _, training_metrics = self.training_function(
            training_data_loader=self.training_data_loader,
            model=self.model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_function=loss_function,
            visualizations_folder=self.visualizations_folder,
        )
        return (
            get_model_parameters(self.model),
            len(self.training_data_loader),
            asdict(training_metrics),
        )

    def evaluate(
        self, parameters, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        print(f"[Client {self.cid}] evaluate, config: {config}")
        set_model_parameters(self.model, parameters)

        f1_score, validation_metrics = self.evaluation_function(
            data_loader=self.validation_data_loader,
            model=self.model,
            loss_function=nn.CrossEntropyLoss(
                weight=self.loss_function_weighting,
            ).to(self.device),
            visualizations_folder=self.visualizations_folder,
        )
        return (
            float(f1_score),
            len(self.validation_data_loader),
            asdict(validation_metrics),
        )

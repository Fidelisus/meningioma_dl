import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Callable

import flwr as fl
import numpy as np
from flwr.common import Scalar, Parameters, EvaluateRes, MetricsAggregationFn, FitIns
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from meningioma_dl.training_utils import _save_model


class SaveModelFedAvg(fl.server.server.FedAvg):
    saved_models_folder: Path
    trained_model_path: Optional[Path] = None
    best_model_f_score: float = 0.0
    last_model: Optional[Parameters] = None

    def aggregate_fit(
        self,
        server_round: int,
        results,
        failures,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        if aggregated_parameters is not None:
            self.last_model = aggregated_parameters

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(
            rnd, results, failures
        )
        if metrics_aggregated["f_score"] > self.best_model_f_score:
            self.best_model_f_score = metrics_aggregated["f_score"]
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_weights(
                self.last_model
            )
            logging.info(
                f"Saving best model with the f_score of {self.best_model_f_score}. "
                f"Saving it at {self.saved_models_folder}"
            )
            self.trained_model_path = _save_model(
                aggregated_ndarrays,
                self.saved_models_folder,
                -1,  # -1 used to override previous best model
            )
        return loss_aggregated, metrics_aggregated


class FedProx(SaveModelFedAvg):
    r"""Federated Optimization strategy.

    Implementation based on https://arxiv.org/abs/1812.06127

    The strategy in itself will not be different than FedAvg, the client needs to
    be adjusted.
    A proximal term needs to be added to the loss function during the training:

    .. math::
        \\frac{\\mu}{2} || w - w^t ||^2

    Where $w^t$ are the global parameters and $w$ are the local weights the function
    will be optimized with.

    In PyTorch, for example, the loss would go from:

    .. code:: python

      loss = criterion(net(inputs), labels)

    To:

    .. code:: python

      for local_weights, global_weights in zip(net.parameters(), global_params):
          proximal_term += (local_weights - global_weights).norm(2)
      loss = criterion(net(inputs), labels) + (config["proximal_mu"] / 2) *
      proximal_term

    With `global_params` being a copy of the parameters before the training takes
    place.

    .. code:: python

      global_params = copy.deepcopy(net).parameters()

    Parameters
    ----------
    fraction_fit : float, optional
        Fraction of clients used during training. In case `min_fit_clients`
        is larger than `fraction_fit * available_clients`, `min_fit_clients`
        will still be sampled. Defaults to 1.0.
    fraction_evaluate : float, optional
        Fraction of clients used during validation. In case `min_evaluate_clients`
        is larger than `fraction_evaluate * available_clients`,
        `min_evaluate_clients` will still be sampled. Defaults to 1.0.
    min_fit_clients : int, optional
        Minimum number of clients used during training. Defaults to 2.
    min_evaluate_clients : int, optional
        Minimum number of clients used during validation. Defaults to 2.
    min_available_clients : int, optional
        Minimum number of total clients in the system. Defaults to 2.
    evaluate_fn : Optional[Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]]
        Optional function used for validation. Defaults to None.
    on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure training. Defaults to None.
    on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure validation. Defaults to None.
    accept_failures : bool, optional
        Whether or not accept rounds containing failures. Defaults to True.
    initial_parameters : Parameters, optional
        Initial global model parameters.
    fit_metrics_aggregation_fn : Optional[MetricsAggregationFn]
        Metrics aggregation function, optional.
    evaluate_metrics_aggregation_fn : Optional[MetricsAggregationFn]
        Metrics aggregation function, optional.
    proximal_mu : float
        The weight of the proximal term used in the optimization. 0.0 makes
        this strategy equivalent to FedAvg, and the higher the coefficient, the more
        regularization will be used (that is, the client parameters will need to be
        closer to the server parameters during training).
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_eval: float = 1.0,
        min_fit_clients: int = 2,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        proximal_mu: float,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_eval=fraction_eval,
            min_fit_clients=min_fit_clients,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        self.proximal_mu = proximal_mu

    def __repr__(self) -> str:
        return (
            f"FedProx(fraction_fit={self.fraction_fit}, min_fit_clients="
            f"{self.min_fit_clients}, proximal_mu={self.proximal_mu})"
        )

    def configure_fit(
        self, rnd: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training.

        Sends the proximal factor mu to the clients
        """
        # Get the standard client/config pairs from the FedAvg super-class
        client_config_pairs = super().configure_fit(rnd, parameters, client_manager)

        # Return client/config pairs with the proximal factor mu added
        return [
            (
                client,
                FitIns(
                    fit_ins.parameters,
                    {**fit_ins.config, "proximal_mu": self.proximal_mu},
                ),
            )
            for client, fit_ins in client_config_pairs
        ]

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Any, Optional, List
from typing import Tuple, Dict

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn import metrics
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
)

from meningioma_dl.experiments_specs.modelling_specs import ModellingSpecs


@dataclass
class TrainingMetrics:
    validation_losses: Sequence[Optional[float]]
    training_losses: Sequence[Optional[float]]
    f_scores: Sequence[Optional[float]]
    learning_rates: Sequence[Optional[float]]


@dataclass
class ValidationMetrics:
    f_score: float
    loss: float
    true: np.array
    predictions: np.array


def calculate_sensitivity_and_specificity(
    true: np.array, predictions: np.array, n_classes: int
) -> Tuple[Dict[str, float], Dict[str, float]]:
    sensitivities = {}
    specificities = {}
    for label in range(n_classes):
        _, recall, _, _ = precision_recall_fscore_support(
            true == label,
            predictions == label,
            pos_label=True,
            average=None,
        )
        sensitivities[str(label)] = recall[1]
        specificities[str(label)] = recall[0]
    return sensitivities, specificities


def create_evaluation_report(
    true: np.array,
    predictions: np.array,
    save_path: Path,
    run_id: str,
    modelling_specs: ModellingSpecs,
    training_specs: Any,
) -> None:
    n_classes = modelling_specs.model_specs.number_of_classes
    report = classification_report(true, predictions, output_dict=True)

    fig = make_subplots(
        rows=3,
        cols=2,
        # shared_xaxes=True,
        vertical_spacing=0.1,
        specs=[
            [{"type": "heatmap"}, {"type": "table"}],
            [{"type": "table", "colspan": 2}, None],
            [{"type": "table", "colspan": 2}, None],
        ],
        subplot_titles=(
            "Confusion matrix",
            "Sensitivity and specificity for each class",
            "F-score, recall and precision",
            "Run specification",
        ),
    )

    fig = add_confusion_matrix_plot(true, predictions, n_classes, fig, row=1, col=1)

    sensitivities, specificities = calculate_sensitivity_and_specificity(
        true, predictions, n_classes
    )
    labels_available = list(range(1, n_classes + 1))
    fig.add_trace(
        go.Table(
            header=dict(
                values=[
                    "Severity (class)",
                    "Sensitivity",
                    "Specificity",
                ],
                font=dict(size=10),
                align="center",
            ),
            cells=dict(
                values=[
                    labels_available,
                    [sensitivities[str(label - 1)] for label in labels_available],
                    [specificities[str(label - 1)] for label in labels_available],
                ],
                align="center",
                format=[None, ".4f", ".4f"],
            ),
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Table(
            header=dict(
                values=["Metric type", "F1 score", "Recall", "Precision"],
                font=dict(size=10),
                align="center",
            ),
            cells=dict(
                values=[
                    ["Macro average", "Macro weighted average"],
                    [
                        report["macro avg"]["f1-score"],
                        report["weighted avg"]["f1-score"],
                    ],
                    [report["macro avg"]["recall"], report["weighted avg"]["recall"]],
                    [
                        report["macro avg"]["precision"],
                        report["weighted avg"]["precision"],
                    ],
                ],
                align="center",
                format=[None, ".4f", ".4f", ".4f"],
            ),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Table(
            header=dict(
                values=["", "Run specification"],
                font=dict(size=10),
                align="left",
            ),
            cells=dict(
                values=[
                    ["Modelling specification", "Training specification"],
                    [str(modelling_specs), str(training_specs)],
                ],
                align="center",
            ),
        ),
        row=3,
        col=1,
    )
    fig.update_layout(
        showlegend=False,
        title_text=f"Run id: {run_id}",
    )

    save_path.mkdir(parents=True, exist_ok=True)
    fig.write_html(save_path.joinpath("evaluation_report.html"))


def add_confusion_matrix_plot(
    true: np.array,
    predictions: np.array,
    n_classes: int,
    fig: go.Figure,
    row: int,
    col: int,
):
    """
    Structure:
            pred1 pred2 pred3
    real1
    real2
    real3
    """
    # It needs to be flipped due to the way how go.Heatmap works
    confusion_matrix_data = np.flip(
        metrics.confusion_matrix(true, predictions).astype(int), axis=0
    )
    x_labels = [f"predicted {label}" for label in range(1, n_classes + 1)]
    y_labels = [f"true {label}" for label in range(n_classes, 0, -1)]

    heatmap = go.Heatmap(
        z=confusion_matrix_data,
        x=x_labels,
        y=y_labels,
        colorscale="Pinkyl",
        colorbar=None,
    )
    fig.add_trace(heatmap, row=row, col=col)

    for i, row_data in enumerate(confusion_matrix_data):
        for j, value in enumerate(row_data):
            fig.add_annotation(
                go.layout.Annotation(
                    x=x_labels[j],
                    y=y_labels[i],
                    text=str(value),
                    showarrow=False,
                ),
                row=row,
                col=col,
            )
    return fig


def plot_training_curve(
    validation_losses: Sequence[float],
    training_losses: Sequence[float],
    f_scores: Sequence[float],
    learning_rates: Sequence[float],
    save_path: Path,
):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(
            "Loss over epochs",
            "F-score and learning rate over epochs",
        ),
    )
    fig.add_trace(
        go.Scatter(
            y=training_losses,
            x=tuple(range(len(training_losses))),
            name="training loss",
            mode="markers+lines",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            y=validation_losses,
            x=tuple(range(len(training_losses))),
            name="validation loss",
            mode="markers+lines",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            y=f_scores,
            x=tuple(range(len(training_losses))),
            name="F1 score",
            mode="markers+lines",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            y=learning_rates,
            x=tuple(range(len(training_losses))),
            name="Learning rate",
            mode="markers+lines",
        ),
        row=2,
        col=1,
    )
    fig.update_layout(title_text="<i><b>Learning curve</b></i>")

    save_path.mkdir(parents=True, exist_ok=True)
    fig.write_html(save_path.joinpath("learning_curve.html"))


def blow_up_metrics_as_3d_matrix(
    quantity: np.ndarray, epochs_in_round: int
) -> np.ndarray:
    """
    blow up from:
    quantity[round][client_id]

    to:
    quantity[round][client_id][epoch_in_round]
    """
    return np.repeat(quantity, epochs_in_round, axis=1).reshape(
        (quantity.shape[0], quantity.shape[1], epochs_in_round)
    )


def get_metric_linear_traces(
    metric: np.ndarray,
    quantity_name: str,
    n_samples_per_client: Optional[np.ndarray] = None,
    main_lines_color: str = "red",
    client_lines_color: str = "brown",
    client_samples_number_lines_color: str = "black",
) -> List[go.Scatter]:
    traces = []

    x_indices = tuple(range(metric.shape[0] * metric.shape[2]))
    traces.append(
        create_scatter_plot_trace(
            y=np.median(metric, axis=1).flatten(),
            x=x_indices,
            name=f"{quantity_name} median",
            color=main_lines_color,
        )
    )
    traces.append(
        create_scatter_plot_trace(
            y=np.quantile(metric, 0.25, axis=1).flatten(),
            x=x_indices,
            name=f"{quantity_name} 25 percentile",
            color=main_lines_color,
            opacity=0.5,
        )
    )
    traces.append(
        create_scatter_plot_trace(
            y=np.quantile(metric, 0.75, axis=1).flatten(),
            x=x_indices,
            name=f"{quantity_name} 75 percentile",
            color=main_lines_color,
            opacity=0.5,
        )
    )
    if n_samples_per_client is not None:
        n_samples_per_client = blow_up_metrics_as_3d_matrix(
            n_samples_per_client, metric.shape[2]
        )

    for client_id in range(metric.shape[1]):
        client_opacity = 0.1 + (1 + client_id) / metric.shape[1] * 0.4
        traces.append(
            create_scatter_plot_trace(
                y=metric[:, client_id, :].flatten(),
                x=x_indices,
                name=f"{quantity_name} for client {client_id}",
                toogled_on=False,
                color=client_lines_color,
                opacity=client_opacity,
            )
        )
        if n_samples_per_client is not None:
            traces.append(
                go.Bar(
                    y=n_samples_per_client[:, client_id].flatten(),
                    x=x_indices,
                    name=f"Samples for a client {client_id}",
                    opacity=client_opacity,
                    visible="legendonly",
                    marker=dict(color=client_samples_number_lines_color),
                )
            )

    return traces


def create_scatter_plot_trace(
    x: np.array,
    y: np.array,
    name: str,
    toogled_on: bool = True,
    color: str = None,
    opacity: float = 1.0,
) -> go.Scatter:
    return go.Scatter(
        y=y,
        x=x,
        name=name,
        mode="markers+lines",
        opacity=opacity,
        visible=None if toogled_on else "legendonly",
        line=dict(color=color),
        connectgaps=True,
    )


def plot_fl_training_curve(
    n_samples_per_client_training: np.ndarray,
    n_samples_per_client_validation: np.ndarray,
    validation_losses: np.ndarray,
    training_losses: np.ndarray,
    f_scores: np.ndarray,
    learning_rates: np.ndarray,
    save_path: Path,
):
    """
    Every quantity should be passed as a matrix with the following structure:
    quantity[round][client_id][epoch_in_round]
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(
            "Loss over epochs",
            "F-score and learning rate over epochs",
        ),
    )
    traces_losses_row: List[go.Scatter] = []
    traces_losses_row.extend(
        get_metric_linear_traces(
            training_losses,
            "Training loss",
            n_samples_per_client=n_samples_per_client_training,
            main_lines_color="red",
            client_lines_color="brown",
            client_samples_number_lines_color="black",
        )
    )
    traces_losses_row.extend(
        get_metric_linear_traces(
            validation_losses,
            "Validation loss",
            n_samples_per_client=n_samples_per_client_validation,
            main_lines_color="green",
            client_lines_color="skyblue",
            client_samples_number_lines_color="blue",
        )
    )
    traces_f_scores_row: List[go.Scatter] = []
    traces_f_scores_row.extend(
        get_metric_linear_traces(
            f_scores,
            "F scores",
            n_samples_per_client=n_samples_per_client_validation,
            main_lines_color="darkgreen",
            client_lines_color="darkslategrey",
            client_samples_number_lines_color="blue",
        )
    )
    traces_f_scores_row.append(
        create_scatter_plot_trace(
            x=tuple(range(learning_rates.shape[0] * learning_rates.shape[2])),
            y=learning_rates[:, 0, :].flatten(),
            name="Learning rate",
            color="grey",
        )
    )

    for trace in traces_losses_row:
        fig.add_trace(trace, row=1, col=1)
    for trace in traces_f_scores_row:
        fig.add_trace(trace, row=2, col=1)

    fig.update_layout(title_text="<i><b>Learning curve</b></i>")

    save_path.mkdir(parents=True, exist_ok=True)
    fig.write_html(save_path.joinpath("learning_curve.html"))

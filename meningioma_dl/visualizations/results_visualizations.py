from pathlib import Path
from typing import Sequence
from typing import Tuple, Dict

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn import metrics
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
)

from meningioma_dl.experiments_specs.experiments import ModellingSpecs
from meningioma_dl.experiments_specs.traning_specs import CentralizedTrainingSpecs


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
    n_classes: int,
    save_path: Path,
    run_id: str,
    modelling_specs: ModellingSpecs,
    training_specs: CentralizedTrainingSpecs,
) -> None:
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
    sensitivities, specificities = calculate_sensitivity_and_specificity(
        true, predictions, n_classes
    )

    fig = add_confusion_matrix_plot(true, predictions, fig, row=1, col=1)

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
                    [1, 2, 3],
                    [sensitivities["0"], sensitivities["1"], sensitivities["2"]],
                    [specificities["0"], specificities["1"], specificities["2"]],
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
    true: np.array, predictions: np.array, fig: go.Figure, row: int, col: int
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
    x_labels = ["predicted 1", "predicted 2", "predicted 3"]
    y_labels = ["real 3", "real 2", "real 1"]

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

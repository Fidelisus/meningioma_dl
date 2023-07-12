from pathlib import Path

import numpy as np
from sklearn import metrics
import plotly.figure_factory as ff


def plot_confusion_matrix(labels: np.array, predictions: np.array, save_path: Path):
    confusion_matrix = metrics.confusion_matrix(labels, predictions)
    confusion_matrix = confusion_matrix.astype(int)
    fig = ff.create_annotated_heatmap(
        confusion_matrix,
        x=["predicted 1", "predicted 2", "predicted 3"],
        y=["real 1", "real 2", "real 3"],
        colorscale="Viridis",
    )

    fig.update_layout(title_text="<i><b>Confusion matrix</b></i>")

    fig["data"][0]["showscale"] = True
    fig.write_html(save_path.joinpath("confusion_matrix.html"))

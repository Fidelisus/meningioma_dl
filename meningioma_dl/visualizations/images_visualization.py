from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from matplotlib import pyplot as plt


def visualize_images(
    data: np.array,
    save_dir: Optional[Path] = None,
    images_names: Optional[Sequence[str]] = None,
):
    if len(images_names) != data.shape[0]:
        raise ValueError(
            f"Number of images {data.shape[0]} should be the same as len of {images_names}"
        )
    for image_with_channel_dimension, image_name in zip(data, images_names):
        image = image_with_channel_dimension[0]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        sagittal_slice = image[:, image.shape[1] // 2, :]
        axes[0].imshow(sagittal_slice, cmap="gray")
        axes[0].set_title("Sagittal Slice")
        axes[0].axis("off")

        coronal_slice = image[image.shape[0] // 2, :, :]
        axes[1].imshow(coronal_slice, cmap="gray")
        axes[1].set_title("Coronal Slice")
        axes[1].axis("off")

        axial_slice = image[:, :, image.shape[2] // 2]
        axes[2].imshow(axial_slice, cmap="gray")
        axes[2].set_title("Axial Slice")
        axes[2].axis("off")

        if save_dir is not None:
            fig.savefig(str(save_dir.joinpath(image_name)))
        else:
            fig.show()

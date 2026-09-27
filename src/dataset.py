from numpy.typing import NDArray
import numpy as np


def train_validation_split(
    images: NDArray[np.uint8],
    labels: NDArray[np.uint8],
    validation_size: int,
    rng: np.random.Generator,
) -> tuple[
    NDArray[np.uint8],
    NDArray[np.uint8],
    NDArray[np.uint8],
    NDArray[np.uint8],
]:
    indices = rng.permutation(len(images))

    validation_indices = indices[:validation_size]
    train_indices = indices[validation_size:]

    validation_images = images[validation_indices]
    validation_labels = labels[validation_indices]

    train_images = images[train_indices]
    train_labels = labels[train_indices]

    return (train_images, train_labels, validation_images, validation_labels)

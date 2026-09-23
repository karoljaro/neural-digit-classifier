from numpy.typing import NDArray
import numpy as np


def prepare_image(raw_image: np.uint8) -> NDArray[np.float32]:
    image: NDArray[np.float32] = np.asarray(
        raw_image,
        dtype=np.float32,
    )

    np.divide(
        image,
        np.float32(255.0),
        out=image,
    )

    return image.reshape(-1, 1)


def one_hot_encode(label: int) -> NDArray[np.float32]:
    one_hot = np.zeros((1, 10), dtype=np.float32)
    one_hot[0, label] = 1.0

    return one_hot

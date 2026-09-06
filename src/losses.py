import numpy as np
from numpy.typing import NDArray


def cross_entropy(
    probabilities: NDArray[np.float32],
    target: NDArray[np.float32],
) -> np.float32:
    log_probabilities = np.log(probabilities)
    loss = -(target @ log_probabilities)

    return np.float32(loss.item())

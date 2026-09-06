from numpy.typing import NDArray
import numpy as np


def relu(z: NDArray[np.float32]) -> NDArray[np.float32]:
    return z.clip(min=0, max=None)

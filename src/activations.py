from numpy.typing import NDArray
import numpy as np


def relu(z: NDArray[np.float32]) -> NDArray[np.float32]:
    return z.clip(min=0, max=None)


def softmax(z: NDArray[np.float32]) -> NDArray[np.float32]:
    shifted = z - np.max(z)
    exp_values = np.exp(shifted)

    sum_of_exps = np.sum(exp_values)

    result = exp_values / sum_of_exps

    return np.asarray(result, dtype=np.float32)

from numpy.typing import NDArray
import numpy as np
from activations import relu


class NeuralNetwork:
    def __init__(self) -> None:
        rng = np.random.default_rng(42)

        self._W1 = rng.normal(loc=0.0, scale=0.01, size=(64, 784)).astype(np.float32)
        self._b1 = np.zeros((64, 1), dtype=np.float32)

        self._W2 = rng.normal(loc=0.0, scale=0.01, size=(10, 64)).astype(np.float32)
        self._b2 = np.zeros((10, 1), dtype=np.float32)

    def forward(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        z1 = self._W1 @ x + self._b1
        a1 = relu(z1)

        z2 = self._W2 @ a1 + self._b2

        return z2

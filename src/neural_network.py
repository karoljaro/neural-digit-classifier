from numpy.typing import NDArray
import numpy as np
from activations import relu, relu_derivative


class NeuralNetwork:
    def __init__(self) -> None:
        rng = np.random.default_rng(42)

        self._W1 = rng.normal(loc=0.0, scale=0.01, size=(64, 784)).astype(np.float32)
        self._b1 = np.zeros((64, 1), dtype=np.float32)

        self._W2 = rng.normal(loc=0.0, scale=0.01, size=(10, 64)).astype(np.float32)
        self._b2 = np.zeros((10, 1), dtype=np.float32)

        self._x: NDArray[np.float32]
        self._z1: NDArray[np.float32]
        self._a1: NDArray[np.float32]

        self._dW1 = np.zeros_like(self._W1, dtype=np.float32)
        self._db1 = np.zeros_like(self._b1, dtype=np.float32)

        self._dW2 = np.zeros_like(self._W2, dtype=np.float32)
        self._db2 = np.zeros_like(self._b2, dtype=np.float32)

    def forward(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        self._x = x
        self._z1 = self._W1 @ x + self._b1
        self._a1 = relu(self._z1)

        z2 = self._W2 @ self._a1 + self._b2

        return z2

    def backward(
        self,
        softmax: NDArray[np.float32],
        one_hot: NDArray[np.float32]
    ) -> None:
        dz2: NDArray[np.float32] = np.asarray(
            softmax - one_hot.T,
            dtype=np.float32
        )

        self._dW2 = dz2 @ self._a1.T
        self._db2 = dz2

        da1 = self._W2.T @ dz2
        dz1: NDArray[np.float32] = np.asarray(
            da1 * relu_derivative(self._z1),
            dtype=np.float32
        )

        self._dW1 = dz1 @ self._x.T
        self._db1 = dz1

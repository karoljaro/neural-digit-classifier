from image import Image
from pathlib import Path
from neural_network import NeuralNetwork
from activations import softmax
from losses import cross_entropy
import numpy as np


def main() -> None:
    image = Image()
    current_dir = Path.cwd()
    file_path = current_dir / "sample" / "one.png"
    loaded_image = image.load(file_path)

    x = loaded_image.reshape(-1, 1)

    neural_network = NeuralNetwork()
    forwarded = neural_network.forward(x)

    softmaxed = softmax(forwarded)

    one_hot = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)

    loss = cross_entropy(softmaxed, one_hot)

    print(loss)


if __name__ == "__main__":
    main()

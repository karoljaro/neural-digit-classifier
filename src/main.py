from pathlib import Path
from neural_network import NeuralNetwork
from activations import softmax
from losses import cross_entropy
import numpy as np
from idx_loader import IdxReader
from numpy.typing import NDArray


def main() -> None:
    reader = IdxReader(
        train_images_path=Path("data/mnist/train-images-idx3-ubyte.gz"),
        train_labels_path=Path("data/mnist/train-labels-idx1-ubyte.gz"),
    )

    images, labels = reader.load()

    neural_network = NeuralNetwork()

    for epoch in range(3):
        total_loss = 0.0
        for raw_image, raw_label in zip(images, labels, strict=True):
            image: NDArray[np.float32] = np.asarray(
                raw_image,
                dtype=np.float32,
            )

            np.divide(
                image,
                np.float32(255.0),
                out=image,
            )

            label = int(raw_label)

            x = image.reshape(-1, 1)
            one_hot = np.zeros((1, 10), dtype=np.float32)
            one_hot[0, label] = 1.0

            forwarded = neural_network.forward(x)
            softmaxed = softmax(forwarded)
            loss = cross_entropy(softmaxed, one_hot)
            total_loss += float(loss)

            neural_network.backward(softmaxed, one_hot)
            neural_network.update_parameters()

        average_loss = total_loss / len(images)
        print(f"Epoch {epoch + 1}: loss={average_loss:.4f}")


if __name__ == "__main__":
    main()

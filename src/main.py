from pathlib import Path
from neural_network import NeuralNetwork
from idx_loader import IdxReader
from training import train_epoch
from evaluation import evaluate_accuracy
import numpy as np
from dataset import train_validation_split


def main() -> None:
    shuffle_rng = np.random.default_rng(42)
    split_rng = np.random.default_rng(43)
    reader = IdxReader(
        images_path=Path("data/mnist/train-images-idx3-ubyte.gz"),
        labels_path=Path("data/mnist/train-labels-idx1-ubyte.gz"),
    )

    test_reader = IdxReader(
        images_path=Path("data/mnist/t10k-images-idx3-ubyte.gz"),
        labels_path=Path("data/mnist/t10k-labels-idx1-ubyte.gz"),
    )

    images, labels = reader.load()
    test_images, test_labels = test_reader.load()
    hidden_size = 32
    print(f"Hidden Size: {hidden_size}")
    neural_network = NeuralNetwork(hidden_size)

    (
        train_images,
        train_labels,
        validation_images,
        validation_labels,
    ) = train_validation_split(
        images,
        labels,
        validation_size=10_000,
        rng=split_rng,
    )

    for epoch in range(10):
        loss = train_epoch(neural_network, train_images, train_labels, shuffle_rng)
        validation_accuracy = evaluate_accuracy(
            neural_network, validation_images, validation_labels
        )

        print(
            f"Epoch {epoch + 1}: "
            f"train_loss={loss:.4f}, "
            f"validation_accuracy={validation_accuracy:.4%}"
        )

    # test_accuracy = evaluate_accuracy(neural_network, test_images, test_labels)

    # print(f"Final test_accuracy={test_accuracy:.4%}")


if __name__ == "__main__":
    main()

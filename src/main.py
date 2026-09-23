from pathlib import Path
from neural_network import NeuralNetwork
from idx_loader import IdxReader
from training import train_epoch
from evaluation import evaluate_accuracy


def main() -> None:
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

    neural_network = NeuralNetwork()

    for epoch in range(3):
        loss = train_epoch(neural_network, images, labels)
        accuracy = evaluate_accuracy(neural_network, test_images, test_labels)

        print(
            f"Epoch {epoch + 1}: "
            f"loss={loss:.4f}, "
            f"test_accuracy={accuracy:.4%}"
        )


if __name__ == "__main__":
    main()

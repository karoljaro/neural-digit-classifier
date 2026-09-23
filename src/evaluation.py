from numpy.typing import NDArray
import numpy as np
from neural_network import NeuralNetwork
from preprocessing import prepare_image
from activations import softmax


def evaluate_accuracy(
    network: NeuralNetwork,
    images: NDArray[np.uint8],
    labels: NDArray[np.uint8],
) -> float:
    correct_count = 0

    for raw_image, raw_label in zip(images, labels, strict=True):
        prepared_image = prepare_image(raw_image)
        label = int(raw_label)

        forwarded = network.forward(prepared_image)
        softmaxed = softmax(forwarded)

        predicted = int(np.argmax(softmaxed))

        if predicted == label:
            correct_count += 1

    accuracy = correct_count / len(images)

    return accuracy

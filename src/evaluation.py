from numpy.typing import NDArray
import numpy as np
from neural_network import NeuralNetwork
from preprocessing import prepare_image
from activations import softmax
from preprocessing import one_hot_encode
from losses import cross_entropy_from_logits


def evaluate(
    network: NeuralNetwork,
    images: NDArray[np.uint8],
    labels: NDArray[np.uint8],
) -> tuple[float, float]:
    correct_count: int = 0
    total_loss: float = 0.0

    for raw_image, raw_label in zip(images, labels, strict=True):
        prepared_image = prepare_image(raw_image)
        label = int(raw_label)

        one_hot = one_hot_encode(label)
        forwarded = network.forward(prepared_image)
        softmaxed = softmax(forwarded)
        loss = cross_entropy_from_logits(forwarded, one_hot)
        total_loss += float(loss)

        predicted = int(np.argmax(softmaxed))

        if predicted == label:
            correct_count += 1

    average_loss = total_loss / len(images)
    accuracy = correct_count / len(images)

    return average_loss, accuracy

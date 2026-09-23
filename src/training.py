from numpy.typing import NDArray
import numpy as np
from preprocessing import prepare_image, one_hot_encode
from neural_network import NeuralNetwork
from activations import softmax
from losses import cross_entropy


def train_epoch(
    network: NeuralNetwork,
    train_images: NDArray[np.uint8],
    train_labels: NDArray[np.uint8],
) -> float:
    total_loss = 0.0

    for raw_image, raw_label in zip(train_images, train_labels, strict=True):
        prepared_image = prepare_image(raw_image)
        label = int(raw_label)

        one_hot = one_hot_encode(label)

        forwarded = network.forward(prepared_image)
        softmaxed = softmax(forwarded)
        loss = cross_entropy(softmaxed, one_hot)
        total_loss += float(loss)

        network.backward(softmaxed, one_hot)
        network.update_parameters()

    average_loss = total_loss / len(train_images)
    return average_loss

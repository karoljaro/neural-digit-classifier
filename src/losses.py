import numpy as np
from numpy.typing import NDArray


def cross_entropy(
    probabilities: NDArray[np.float32],
    target: NDArray[np.float32],
) -> np.float32:
    log_probabilities = np.log(probabilities)
    loss = -(target @ log_probabilities)

    return np.float32(loss.item())


def cross_entropy_from_logits(
    logits: NDArray[np.float32],
    target: NDArray[np.float32]
) -> np.float32:
    # Stabilna numerycznie wersja cross-entropy liczona bezpośrednio z logits.
    #
    # Klasyczne podejście:
    #   logits -> softmax -> probabilities -> log(probabilities)
    #
    # może prowadzić do problemu, gdy softmax zwróci bardzo małe wartości,
    # które w float32 zostaną zaokrąglone do 0. Wtedy log(0) daje -inf,
    # a dalsze obliczenia mogą zakończyć się wartością nan.
    #
    # Tutaj wykorzystujemy równoważne przekształcenie matematyczne:
    # log(softmax(z)) = shifted - log(sum(exp(shifted))),
    # gdzie shifted = z - max(z).
    #
    # Dzięki temu nie musimy najpierw tworzyć bardzo małych probability
    # i unikamy underflow oraz log(0).
    shifted = logits - np.max(logits)

    log_sum_exp = np.log(
        np.sum(np.exp(shifted))
    )

    log_probabilities = shifted - log_sum_exp
    loss = -(target @ log_probabilities)

    return np.float32(loss.item())

from typing import Final


class EarlyStopping:
    def __init__(self) -> None:
        self._best_validation_loss: float | None = None

        self._bad_counter: int = 0
        self._no_improvement: int = 0

        self._GL_THRESHOLD: Final[float] = 5.0  # 5%
        self._BAD_PATIENCE: Final[int] = 4
        self._MAX_NO_IMPROVEMENT: Final[int] = 10

    def should_stop(self, latest_validation_loss: float) -> bool:
        if (
            self._best_validation_loss is None
            or latest_validation_loss < self._best_validation_loss
        ):
            self._best_validation_loss = latest_validation_loss
            self._bad_counter = 0
            self._no_improvement = 0
            return False

        gl = self._calculate_generalization_loss(latest_validation_loss)

        self._no_improvement += 1
        if self._no_improvement >= self._MAX_NO_IMPROVEMENT:
            return True

        if gl <= self._GL_THRESHOLD:
            self._bad_counter = 0
            return False

        self._bad_counter += 1
        if self._bad_counter >= self._BAD_PATIENCE:
            return True

        return False

    def _calculate_generalization_loss(self, current_validation_loss: float) -> float:
        if self._best_validation_loss is None:
            raise RuntimeError("Best validation loss has not been initialized")
        return ((current_validation_loss / self._best_validation_loss) - 1) * 100

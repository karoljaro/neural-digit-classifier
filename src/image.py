from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray


class Image:
    def load(self, path: Path) -> NDArray[np.float32]:
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        image = cv2.imread(str(path))

        if image is None:
            raise ValueError(f"Failed to load image: {path}")

        grayscaled_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(grayscaled_image, (28, 28))
        image_float = resized_image.astype(np.float32)

        img_min = image_float.min()
        img_max = image_float.max()

        if img_max - img_min > 0:
            normalized = (image_float - img_min) / (img_max - img_min)
            normalized = 1.0 - normalized
        else:
            normalized = np.zeros_like(image_float)

        return normalized

from pathlib import Path
import gzip

import numpy as np
from numpy.typing import NDArray


class IdxReader:
    IMAGE_MAGIC = 2051
    LABEL_MAGIC = 2049

    def __init__(
        self,
        train_images_path: Path,
        train_labels_path: Path,
    ) -> None:
        self._train_images_path = train_images_path
        self._train_labels_path = train_labels_path

    def load(self) -> tuple[NDArray[np.uint8], int]:
        self._validate_paths()

        image, image_count = self._load_first_image()
        label, label_count = self._load_first_label()

        if image_count != label_count:
            raise ValueError(
                f"Image count ({image_count}) does not match "
                f"label count ({label_count})"
            )

        return image, label

    def _load_first_image(self) -> tuple[NDArray[np.uint8], int]:
        with gzip.open(self._train_images_path, "rb") as file:
            header = file.read(16)

            magic = int.from_bytes(header[0:4], byteorder="big")
            image_count = int.from_bytes(header[4:8], byteorder="big")
            rows = int.from_bytes(header[8:12], byteorder="big")
            columns = int.from_bytes(header[12:16], byteorder="big")

            if magic != self.IMAGE_MAGIC:
                raise ValueError(f"Invalid image magic number: {magic}")

            image_bytes = file.read(rows * columns)

            flat_image = np.frombuffer(
                image_bytes,
                dtype=np.uint8,
            )

            image = flat_image.reshape(rows, columns)

            return image, image_count

    def _load_first_label(self) -> tuple[int, int]:
        with gzip.open(self._train_labels_path, "rb") as file:
            header = file.read(8)

            magic = int.from_bytes(header[0:4], byteorder="big")
            label_count = int.from_bytes(header[4:8], byteorder="big")

            if magic != self.LABEL_MAGIC:
                raise ValueError(f"Invalid label magic number: {magic}")

            label_byte = file.read(1)
            label = int.from_bytes(label_byte, byteorder="big")

            return label, label_count

    def _validate_paths(self) -> None:
        if not self._train_images_path.is_file():
            raise FileNotFoundError(f"Image file not found: {self._train_images_path}")

        if not self._train_labels_path.is_file():
            raise FileNotFoundError(f"Label file not found: {self._train_labels_path}")

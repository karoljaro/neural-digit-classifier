from pathlib import Path
import gzip
import numpy as np


class IdxReader:
    def __init__(self, train_images_path: Path, train_labels_path: Path) -> None:
        self._train_images_path: Path = train_images_path
        self._train_labels_path: Path = train_labels_path

    def load(self) -> None:
        if (
            not self._train_images_path.is_file()
            or not self._train_labels_path.is_file()
        ):
            raise FileNotFoundError(
                f"File in path: {self._train_images_path} not found"
            )

        with gzip.open(self._train_images_path, "rb") as b_file:
            header = b_file.read(16)

            magic = int.from_bytes(header[0:4], byteorder="big")
            image_count = int.from_bytes(header[4:8], byteorder="big")
            rows = int.from_bytes(header[8:12], byteorder="big")
            columns = int.from_bytes(header[12:16], byteorder="big")

            print(f"magic: {magic}")
            print(f"image count: {image_count}")
            print(f"rows: {rows}")
            print(f"columns: {columns}")

            image_bytes = b_file.read(rows * columns)
            print(len(image_bytes))

            flat_image = np.frombuffer(image_bytes, dtype=np.uint8)
            image = flat_image.reshape(rows, columns)

            print(image.shape)
            print(image)

        with gzip.open(self._train_labels_path, "rb") as b_file:
            h

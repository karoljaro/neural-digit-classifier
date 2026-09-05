from image import Image
from pathlib import Path
import numpy as np


def main() -> None:
    image = Image()
    current_dir = Path.cwd()
    file_path = current_dir / "sample" / "one.png"
    loaded_image = image.load(file_path)

    x = loaded_image.reshape(-1, 1)

    rng = np.random.default_rng(42)

    W = rng.normal(
        loc=0.0,
        scale=0.01,
        size=(64, 784)
    ).astype(np.float32)

    b = np.zeros((64, 1), dtype=np.float32)

    z = W @ x + b
    print(z)


if __name__ == "__main__":
    main()

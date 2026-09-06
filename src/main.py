from image import Image
from pathlib import Path
from neural_network import NeuralNetwork


def main() -> None:
    image = Image()
    current_dir = Path.cwd()
    file_path = current_dir / "sample" / "one.png"
    loaded_image = image.load(file_path)

    x = loaded_image.reshape(-1, 1)

    neural_network = NeuralNetwork()
    forwarded = neural_network.forward(x)

    print(forwarded)


if __name__ == "__main__":
    main()

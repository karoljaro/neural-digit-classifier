from image import Image
from pathlib import Path


def main() -> None:
    image = Image()
    current_dir = Path.cwd()
    file_path = current_dir / "sample" / "one.png"
    print(image.load(file_path))


if __name__ == "__main__":
    main()

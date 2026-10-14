"""Predict a handwritten digit from an image file."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

if __package__:
    from src import tensorflow_config  # noqa: F401
else:
    import tensorflow_config  # noqa: F401

from tensorflow.keras.models import load_model

if __package__:
    from src.data_loader import PROJECT_ROOT
else:
    from data_loader import PROJECT_ROOT


MODEL_PATH = PROJECT_ROOT / "models" / "mnist_model.keras"


def prepare_image(image_source: str | Path | Image.Image) -> np.ndarray:
    """Crop and center a digit in a normalized MNIST-style input tensor."""
    if isinstance(image_source, Image.Image):
        image = image_source.copy()
    else:
        image_path = Path(image_source)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        image = Image.open(image_path)

    pixels = np.asarray(ImageOps.grayscale(image), dtype=np.float32)

    # MNIST uses light digits on a dark background; invert common scanned images.
    border = np.concatenate(
        (pixels[0, :], pixels[-1, :], pixels[:, 0], pixels[:, -1])
    )
    if float(np.mean(border)) > 127:
        pixels = 255.0 - pixels

    foreground = pixels > max(20.0, float(pixels.max()) * 0.1)
    rows, columns = np.where(foreground)
    if rows.size == 0:
        return np.zeros((1, 28, 28, 1), dtype=np.float32)

    digit = Image.fromarray(pixels).crop(
        (columns.min(), rows.min(), columns.max() + 1, rows.max() + 1)
    )
    scale = 20 / max(digit.size)
    digit = digit.resize(
        (
            max(1, round(digit.width * scale)),
            max(1, round(digit.height * scale)),
        ),
        Image.Resampling.LANCZOS,
    )

    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(digit, ((28 - digit.width) // 2, (28 - digit.height) // 2))

    # MNIST digits are centered by their ink mass, not just their bounding box.
    pixels = np.asarray(canvas, dtype=np.float32)
    total = float(pixels.sum())
    if total:
        y_coordinates, x_coordinates = np.indices(pixels.shape)
        center_x = float((pixels * x_coordinates).sum() / total)
        center_y = float((pixels * y_coordinates).sum() / total)
        shift_x = round(13.5 - center_x)
        shift_y = round(13.5 - center_y)
        centered = Image.new("L", (28, 28), 0)
        centered.paste(canvas, (shift_x, shift_y))
        pixels = np.asarray(centered, dtype=np.float32)

    return (pixels / 255.0)[np.newaxis, ..., np.newaxis]


def predict_image(
    image_source: str | Path | Image.Image, model_path: str | Path = MODEL_PATH
) -> tuple[int, float, np.ndarray]:
    """Return the predicted digit, its probability, and all class probabilities."""
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. Run `python -m src.train` first."
        )
    model = load_model(model_path)
    probabilities = model.predict(prepare_image(image_source), verbose=0)[0]
    digit = int(np.argmax(probabilities))
    return digit, float(probabilities[digit]), probabilities


def main() -> None:
    """Parse an image path and print the prediction and confidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_path", type=Path, help="Path to an image of a digit")
    parser.add_argument(
        "--model", type=Path, default=MODEL_PATH, help="Path to a saved Keras model"
    )
    arguments = parser.parse_args()
    digit, confidence, _ = predict_image(arguments.image_path, arguments.model)
    print(f"Predicted digit: {digit}")
    print(f"Confidence: {confidence:.2%}")


if __name__ == "__main__":
    main()

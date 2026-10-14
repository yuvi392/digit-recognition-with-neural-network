"""Download the MNIST dataset and save a small visual sample."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if __package__:
    from src import tensorflow_config  # noqa: F401
else:
    import tensorflow_config  # noqa: F401

from tensorflow.keras.datasets import mnist


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"


def load_mnist_data() -> tuple[
    tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]
]:
    """Load MNIST, report its dimensions, and save a grid of training images."""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(f"Training images: {x_train.shape}; training labels: {y_train.shape}")
    print(f"Test images:     {x_test.shape}; test labels:     {y_test.shape}")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(3, 5, figsize=(9, 6))
    for index, axis in enumerate(axes.flat):
        axis.imshow(x_train[index], cmap="gray")
        axis.set_title(f"Label: {y_train[index]}")
        axis.axis("off")
    figure.suptitle("Sample handwritten digits from MNIST")
    figure.tight_layout()
    figure.savefig(PLOTS_DIR / "sample_images.png", dpi=150)
    plt.close(figure)

    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    load_mnist_data()

"""Prepare MNIST images and labels for neural-network training."""

import numpy as np
from sklearn.model_selection import train_test_split

if __package__:
    from src import tensorflow_config  # noqa: F401
else:
    import tensorflow_config  # noqa: F401

from tensorflow.keras.utils import to_categorical


def normalize_images(images: np.ndarray) -> np.ndarray:
    """Scale grayscale images to 0–1 and add a channel dimension if needed."""
    images = images.astype("float32") / 255.0
    if images.ndim == 3:
        images = np.expand_dims(images, axis=-1)
    return images


def encode_labels(labels: np.ndarray) -> np.ndarray:
    """Convert digit labels 0–9 into ten-element one-hot vectors."""
    return to_categorical(labels, num_classes=10)


def preprocess_data(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    validation_size: float = 0.1,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Normalize images, add a channel dimension, one-hot encode, and split."""
    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be greater than 0 and less than 1.")

    x_train = normalize_images(x_train)
    x_test = normalize_images(x_test)
    y_train = encode_labels(y_train)
    y_test = encode_labels(y_test)
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_train,
        y_train,
        test_size=validation_size,
        random_state=random_state,
        stratify=np.argmax(y_train, axis=1),
    )
    return x_train, y_train, x_validation, y_validation, x_test, y_test

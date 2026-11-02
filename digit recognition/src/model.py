"""Define the baseline dense network and an optional convolutional model."""

if __package__:
    from src import tensorflow_config  # noqa: F401
else:
    import tensorflow_config  # noqa: F401

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D


def build_model(
    model_type: str = "mlp", input_shape: tuple[int, int, int] = (28, 28, 1)
) -> Sequential:
    """Create and compile an MNIST classifier ('mlp' or optional 'cnn')."""
    if model_type == "mlp":
        model = Sequential(
            [
                Flatten(input_shape=input_shape),
                Dense(128, activation="relu"),
                Dropout(0.2),
                Dense(64, activation="relu"),
                Dense(10, activation="softmax"),
            ],
            name="mnist_mlp",
        )
    elif model_type == "cnn":
        model = Sequential(
            [
                Conv2D(32, kernel_size=3, activation="relu", input_shape=input_shape),
                MaxPooling2D(pool_size=2),
                Conv2D(64, kernel_size=3, activation="relu"),
                MaxPooling2D(pool_size=2),
                Flatten(),
                Dense(64, activation="relu"),
                Dropout(0.25),
                Dense(10, activation="softmax"),
            ],
            name="mnist_cnn",
        )
    else:
        raise ValueError("model_type must be either 'mlp' or 'cnn'.")

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

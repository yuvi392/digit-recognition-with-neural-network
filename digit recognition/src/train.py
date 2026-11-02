"""Train an MNIST classifier, plot its learning curves, and save the model."""

import argparse

import matplotlib.pyplot as plt

if __package__:
    from src import tensorflow_config  # noqa: F401
else:
    import tensorflow_config  # noqa: F401

from tensorflow.keras.callbacks import EarlyStopping, History
from tensorflow.keras.models import Sequential

if __package__:
    from src.data_loader import PROJECT_ROOT, load_mnist_data
    from src.model import build_model
    from src.preprocess import preprocess_data
else:
    from data_loader import PROJECT_ROOT, load_mnist_data
    from model import build_model
    from preprocess import preprocess_data


PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"
MODEL_PATH = PROJECT_ROOT / "models" / "mnist_model.keras"


def plot_training_history(history: History) -> None:
    """Save accuracy and loss curves from a completed training run."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history.history["accuracy"]) + 1)

    for metric, title in (("accuracy", "Accuracy"), ("loss", "Loss")):
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.plot(epochs, history.history[metric], label="Training")
        axis.plot(epochs, history.history[f"val_{metric}"], label="Validation")
        axis.set(title=f"Training and validation {title.lower()}", xlabel="Epoch")
        axis.set_ylabel(title)
        axis.legend()
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(PLOTS_DIR / f"{metric}_curve.png", dpi=150)
        plt.close(figure)


def train_model(
    model_type: str = "mlp", epochs: int = 15, batch_size: int = 128
) -> Sequential:
    """Load and prepare MNIST, train with early stopping, then save the model."""
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must both be positive integers.")

    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    x_train, y_train, x_val, y_val, _, _ = preprocess_data(
        x_train, y_train, x_test, y_test
    )
    model = build_model(model_type=model_type)
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=2,
    )

    plot_training_history(history)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")
    return model


def main() -> None:
    """Parse command-line options and start model training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("mlp", "cnn"), default="mlp")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    arguments = parser.parse_args()
    train_model(arguments.model, arguments.epochs, arguments.batch_size)


if __name__ == "__main__":
    main()

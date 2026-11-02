"""Evaluate the saved classifier and write diagnostic plots."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

if __package__:
    from src import tensorflow_config  # noqa: F401
else:
    import tensorflow_config  # noqa: F401

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

if __package__:
    from src.data_loader import PROJECT_ROOT, load_mnist_data
    from src.preprocess import normalize_images
else:
    from data_loader import PROJECT_ROOT, load_mnist_data
    from preprocess import normalize_images


MODEL_PATH = PROJECT_ROOT / "models" / "mnist_model.keras"
PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"


def evaluate_model(model_path: str | Path = MODEL_PATH) -> None:
    """Print test metrics and save a confusion matrix and error examples."""
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. Run `python -m src.train` first."
        )

    (_, _), (x_test, y_test_raw) = load_mnist_data()
    x_test = normalize_images(x_test)
    y_test = to_categorical(y_test_raw, num_classes=10)
    model = load_model(model_path)
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    probabilities = model.predict(x_test, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    actual = np.argmax(y_test, axis=1)

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f} ({test_accuracy:.2%})")
    print(
        classification_report(
            actual, predictions, labels=list(range(10)), zero_division=0
        )
    )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(actual, predictions, labels=list(range(10)))
    figure, axis = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=range(10),
        yticklabels=range(10),
        ax=axis,
    )
    axis.set(title="MNIST confusion matrix", xlabel="Predicted digit", ylabel="True digit")
    figure.tight_layout()
    figure.savefig(PLOTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(figure)

    incorrect_indices = np.flatnonzero(predictions != actual)[:15]
    if incorrect_indices.size:
        figure, axes = plt.subplots(3, 5, figsize=(10, 7))
        for axis, index in zip(axes.flat, incorrect_indices):
            axis.imshow(x_test[index].squeeze(), cmap="gray")
            axis.set_title(f"True: {actual[index]} | Pred: {predictions[index]}")
            axis.axis("off")
        for axis in axes.flat[incorrect_indices.size :]:
            axis.axis("off")
        figure.suptitle("Examples the model misclassified")
        figure.tight_layout()
        figure.savefig(PLOTS_DIR / "misclassified_examples.png", dpi=150)
        plt.close(figure)
        print(
            f"Saved {incorrect_indices.size} misclassified examples to "
            f"{PLOTS_DIR / 'misclassified_examples.png'}"
        )
    else:
        print("No test examples were misclassified.")
    print(f"Saved confusion matrix to {PLOTS_DIR / 'confusion_matrix.png'}")


if __name__ == "__main__":
    evaluate_model()

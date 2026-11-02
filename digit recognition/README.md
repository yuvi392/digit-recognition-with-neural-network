# Handwritten Digit Recognition Using Neural Networks

A beginner-friendly Python project that classifies handwritten digits from the
[MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database) with TensorFlow
and Keras. The default model is a fully connected neural network; an optional
convolutional neural network (CNN) is also included.

## Project overview

The project demonstrates loading and exploring image data, normalizing pixels,
one-hot encoding labels, creating a validation split, training with early
stopping, evaluating classification metrics, and making predictions from new
image files. A Streamlit interface supports both image uploads and freehand
drawing.

The first run downloads MNIST through `keras.datasets.mnist`: 60,000 training
images and 10,000 test images, each a 28 × 28 grayscale image representing a
digit from 0 to 9.

The project suppresses routine TensorFlow startup and GPU-availability warning
messages to keep the beginner-facing console output readable; actual errors
remain visible.

## Project structure

```text
data/                 Reserved for local datasets (MNIST is downloaded by Keras)
models/               Saved Keras model
notebooks/            Guided Jupyter notebook
src/                  Data, preprocessing, training, evaluation, and prediction
app/                  Streamlit app
outputs/plots/        Sample images, learning curves, and evaluation plots
outputs/screenshots/  Reserved for app screenshots
docs/                 Reserved for supplementary documentation
```

## Setup

Python 3.10–3.12 is recommended. From the project root, create and activate a
virtual environment, then install the dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

In Windows PowerShell, if script activation is restricted, use
`.\.venv\Scripts\activate.bat` from Command Prompt instead. TensorFlow may need
a few minutes to install. The MNIST files are downloaded automatically the
first time the data loader runs and are cached by Keras outside this project.

## Run the project

Run all commands from the project root.

1. Download and inspect MNIST; save a labeled image grid:

   ```powershell
   python src/data_loader.py
   ```

2. Train the default dense neural network (15 epochs maximum; early stopping
   may stop sooner):

   ```powershell
   python src/train.py
   ```

   The trained model is saved to `models/mnist_model.keras`. Accuracy and loss
   curves are saved under `outputs/plots/`.

3. Optionally train the CNN version:

   ```powershell
   python src/train.py --model cnn
   ```

   This saves to the same model path, replacing the previous model.

4. Evaluate the saved model on the untouched test set:

   ```powershell
   python src/evaluate.py
   ```

   The terminal shows test loss, accuracy, precision, recall, F1-score, and a
   per-digit classification report. The confusion matrix and misclassified
   image examples are saved under `outputs/plots/`.

5. Predict one image (best results come from one centered digit with a clear
   background):

   ```powershell
   python src/predict.py "path\to\your_digit.png"
   ```

   Supported image formats include PNG, JPG/JPEG, and BMP. The script converts
   the image to grayscale, crops and centers the digit, scales it to fit a
   20 × 20 area within a 28 × 28 image, and inverts bright-background images to
   match MNIST's light-digit-on-dark-background convention.
   To try a real MNIST example included in `data/`, run:

   ```powershell
   python src/predict.py "data\sample_digit.png"
   ```

   You can also try the generated handwritten-style examples for digits 0–9:

   ```powershell
   python src/predict.py "data\generated_digits\digit_7.png"
   ```

   These are synthetic examples, so model predictions may vary.

6. Start the interactive app:

   ```powershell
   streamlit run app/app.py
   ```

   In VS Code, select **Streamlit: MNIST Digit Recognizer** in **Run and
   Debug** and press **F5**. Do not use **Run Python File** for this app.

   Open the local URL printed by Streamlit, then upload an image or select
   **Draw a digit**. The app displays the predicted digit, its confidence, and
   probabilities for all ten classes.

7. Open the guided notebook:

   ```powershell
   jupyter notebook notebooks/mnist_digit_classification.ipynb
   ```

## Expected results

The dense baseline should typically exceed 95% test accuracy after training;
the exact result depends on the runtime, model choice, and training settings.
The CNN option often performs better. The validation set is drawn only from the
training data; the test set is reserved for final evaluation.

## Useful options

Change the epoch limit and batch size when training:

```powershell
python src/train.py --model mlp --epochs 15 --batch-size 128
```

Use a different saved model location for prediction:

```powershell
python src/predict.py "path\to\your_digit.png" --model "models\mnist_model.keras"
```

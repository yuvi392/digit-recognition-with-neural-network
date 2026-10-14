"""Streamlit interface for uploading or drawing a handwritten digit."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from src import tensorflow_config  # noqa: F401

from tensorflow.keras.models import load_model

from src.data_loader import PROJECT_ROOT
from src.predict import prepare_image


MODEL_PATH = PROJECT_ROOT / "models" / "mnist_model.keras"


@st.cache_resource
def get_model():
    """Load the trained model once and reuse it across Streamlit reruns."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Train it with `python -m src.train`."
        )
    return load_model(MODEL_PATH)


st.set_page_config(
    page_title="MNIST Digit Recognizer",
    page_icon="🔢",
    layout="wide",
)
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background: #f4f7fc;
    }
    [data-testid="stHeader"] {
        background: transparent;
    }
    .main .block-container {
        max-width: 1120px;
        padding-top: 2.25rem;
        padding-bottom: 3rem;
    }
    .hero {
        position: relative;
        overflow: hidden;
        padding: 2.1rem 2.4rem;
        border-radius: 24px;
        color: white;
        background: linear-gradient(120deg, #172554 0%, #3730a3 55%, #6d28d9 100%);
        box-shadow: 0 18px 45px rgba(49, 46, 129, 0.2);
    }
    .hero::after {
        content: "123";
        position: absolute;
        right: 2rem;
        top: 0.45rem;
        color: rgba(255, 255, 255, 0.08);
        font-size: 7rem;
        font-weight: 800;
        letter-spacing: -0.12em;
        pointer-events: none;
    }
    .hero-kicker {
        margin-bottom: 0.55rem;
        color: #c7d2fe;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.13em;
        text-transform: uppercase;
    }
    .hero h1 {
        margin: 0;
        color: white;
        font-size: clamp(2rem, 5vw, 3rem);
        font-weight: 800;
        letter-spacing: -0.045em;
    }
    .hero p {
        max-width: 620px;
        margin: 0.65rem 0 0;
        color: #e0e7ff;
        font-size: 1.03rem;
    }
    .section-title {
        margin: 1.6rem 0 0.15rem;
        color: #18223b;
        font-size: 1.2rem;
        font-weight: 750;
    }
    .section-hint {
        margin: 0 0 0.8rem;
        color: #64748b;
        font-size: 0.92rem;
    }
    [data-testid="stRadio"] > label {
        color: #26334d;
        font-weight: 650;
    }
    [data-testid="stRadioOption"] p {
        color: #26334d !important;
    }
    [data-testid="stFileUploader"] section {
        border: 1px dashed #a5b4fc;
        border-radius: 16px;
        background: #f8faff;
    }
    iframe[title="streamlit_drawable_canvas.st_canvas"] {
        width: 282px !important;
    }
    .result-card {
        display: flex;
        align-items: center;
        gap: 1.1rem;
        margin: 0.8rem 0 1.1rem;
        padding: 1.2rem 1.4rem;
        border: 1px solid #e0e7ff;
        border-radius: 20px;
        background: linear-gradient(135deg, #eef2ff, #faf5ff);
    }
    .result-digit {
        display: grid;
        width: 82px;
        height: 82px;
        flex: 0 0 82px;
        place-items: center;
        border-radius: 20px;
        color: white;
        background: linear-gradient(145deg, #4f46e5, #7c3aed);
        box-shadow: 0 10px 22px rgba(79, 70, 229, 0.24);
        font-size: 3.2rem;
        font-weight: 800;
        line-height: 1;
    }
    .result-label {
        margin: 0 0 0.25rem;
        color: #64748b;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .result-value {
        margin: 0;
        color: #1e1b4b;
        font-size: 1.35rem;
        font-weight: 750;
    }
    .result-confidence {
        margin: 0.25rem 0 0;
        color: #475569;
        font-size: 0.95rem;
    }
    .empty-card {
        margin-top: 0.8rem;
        padding: 2.2rem 1.4rem;
        border: 1px solid #e2e8f0;
        border-radius: 20px;
        background: white;
        text-align: center;
    }
    .empty-icon {
        font-size: 2.5rem;
    }
    .empty-card h3 {
        margin: 0.65rem 0 0.35rem;
        color: #1e293b;
        font-size: 1.15rem;
    }
    .empty-card p {
        margin: 0;
        color: #64748b;
        font-size: 0.93rem;
    }
    .footer-note {
        margin-top: 1.3rem;
        color: #64748b;
        font-size: 0.83rem;
        text-align: center;
    }
    </style>
    <div class="hero">
        <div class="hero-kicker">A little machine learning magic</div>
        <h1>What digit do you see?</h1>
        <p>Write a number or upload an image. The neural network will take its best guess.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

input_column, result_column = st.columns([1.05, 0.95], gap="large")
image: Image.Image | None = None

with input_column:
    st.markdown('<div class="section-title">✍️ Give it a digit</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-hint">Draw one number or choose an image from your device.</div>',
        unsafe_allow_html=True,
    )
    input_method = st.radio(
        "Choose an input method",
        ("Draw a digit", "Upload an image"),
        horizontal=True,
        label_visibility="collapsed",
    )

    if input_method == "Upload an image":
        uploaded_file = st.file_uploader(
            "Choose a digit image", type=("png", "jpg", "jpeg", "bmp")
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Your image", width=220)
    else:
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=14,
            stroke_color="#111827",
            background_color="#FFFFFF",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="digit_canvas",
        )
        if canvas_result.image_data is not None:
            image = Image.fromarray(canvas_result.image_data.astype("uint8"), mode="RGBA")

with result_column:
    st.markdown('<div class="section-title">✨ Your prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-hint">The model compares your digit with patterns it learned from MNIST.</div>',
        unsafe_allow_html=True,
    )

    if image is None:
        st.markdown(
            """
            <div class="empty-card">
                <div class="empty-icon">🔢</div>
                <h3>Your result will appear here</h3>
                <p>Start drawing or upload an image to see the prediction.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        try:
            model = get_model()
            model_input = prepare_image(image)
            if not model_input.any():
                st.markdown(
                    """
                    <div class="empty-card">
                        <div class="empty-icon">✍️</div>
                        <h3>No digit detected yet</h3>
                        <p>Draw a dark digit on the canvas, or upload an image with one clear digit.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                probabilities = model.predict(model_input, verbose=0)[0]
                predicted_digit = int(probabilities.argmax())
                confidence = float(probabilities[predicted_digit])
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-digit">{predicted_digit}</div>
                        <div>
                            <p class="result-label">Best match</p>
                            <p class="result-value">Digit {predicted_digit}</p>
                            <p class="result-confidence">Confidence: {confidence:.1%}</p>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown("**Probability for each digit**")
                chart_data = pd.DataFrame(
                    {"Probability": probabilities},
                    index=[str(digit) for digit in range(10)],
                )
                st.bar_chart(chart_data, color="#6366f1", height=280)
        except FileNotFoundError as error:
            st.error(str(error))

st.markdown(
    '<div class="footer-note">For best results, draw one large digit and keep it near the center.</div>',
    unsafe_allow_html=True,
)

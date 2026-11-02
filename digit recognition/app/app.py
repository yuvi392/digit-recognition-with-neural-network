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
from src.predict import prepare_image

MODEL_PATH = PROJECT_ROOT / "models" / "mnist_model.keras"


@st.cache_resource(show_spinner=False)
def get_model():
    """Load the trained model once and reuse it across Streamlit reruns."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Train it with `python -m src.train`."
        )
    return load_model(MODEL_PATH)


st.set_page_config(page_title="Ink Lab | Digit Recognition", page_icon="✦", layout="wide")

st.markdown(
    """
    <style>
    :root { --bg:#101315; --panel:#181d20; --panel-2:#20272a; --ink:#edf2ef; --muted:#98a7a2; --line:#2c3738; --lime:#d6f36b; --orange:#ff9b62; }
    [data-testid="stAppViewContainer"] { background:var(--bg); color:var(--ink); }
    [data-testid="stHeader"] { background:transparent; }
    .main .block-container { max-width:1280px; padding:1.7rem 2.8rem 3rem; }
    .nav { display:flex; align-items:center; justify-content:space-between; padding-bottom:1.5rem; border-bottom:1px solid var(--line); }
    .logo { display:flex; align-items:center; gap:.7rem; color:var(--ink); font-size:1rem; font-weight:850; letter-spacing:-.02em; }
    .logo-mark { display:grid; width:2rem; height:2rem; place-items:center; border-radius:8px; color:#101315; background:var(--lime); font-weight:900; }
    .nav-meta { color:var(--muted); font-size:.75rem; letter-spacing:.1em; text-transform:uppercase; }
    .intro { display:flex; align-items:end; justify-content:space-between; gap:2rem; padding:2.8rem 0 2rem; }
    .eyebrow { margin:0 0 .8rem; color:var(--lime); font-size:.72rem; font-weight:850; letter-spacing:.18em; text-transform:uppercase; }
    .intro h1 { max-width:700px; margin:0; color:var(--ink); font-size:clamp(2.3rem,5vw,4.6rem); font-weight:850; line-height:.98; letter-spacing:-.07em; }
    .intro-copy { max-width:290px; margin:0 0 .2rem; color:var(--muted); font-size:.9rem; line-height:1.6; }
    .workspace { padding:1.3rem; border:1px solid var(--line); border-radius:20px; background:var(--panel); }
    .workspace-label { display:flex; justify-content:space-between; margin-bottom:1rem; color:var(--muted); font-size:.72rem; font-weight:800; letter-spacing:.12em; text-transform:uppercase; }
    .workspace-label span:last-child { color:var(--lime); }
    .input-help { margin:.3rem 0 1rem; color:var(--muted); font-size:.86rem; }
    [data-testid="stRadio"] > label { color:var(--ink); font-weight:750; }
    [data-testid="stRadioOption"] p { color:var(--ink) !important; }
    [data-testid="stFileUploader"] section { border:1px dashed #53635e; border-radius:12px; background:#151a1c; }
    iframe[title="streamlit_drawable_canvas.st_canvas"] { width:282px !important; border:1px solid #52615e; border-radius:12px; box-shadow:0 12px 30px rgba(0,0,0,.2); }
    .insight { min-height:100%; padding:1.5rem; border:1px solid var(--line); border-radius:20px; background:linear-gradient(145deg,#20282a,#171c1e); }
    .insight-label { margin:0 0 1rem; color:var(--muted); font-size:.72rem; font-weight:800; letter-spacing:.12em; text-transform:uppercase; }
    .waiting { display:flex; min-height:200px; flex-direction:column; justify-content:center; border:1px dashed #465452; border-radius:14px; padding:1.5rem; }
    .waiting-symbol { color:#53635e; font-size:3rem; line-height:1; }
    .waiting h3 { margin:.8rem 0 .4rem; color:var(--ink); font-size:1.2rem; }
    .waiting p { max-width:280px; margin:0; color:var(--muted); font-size:.86rem; line-height:1.5; }
    .answer { display:flex; align-items:center; gap:1rem; padding-bottom:1.3rem; border-bottom:1px solid var(--line); }
    .answer-digit { display:grid; width:84px; height:84px; place-items:center; border-radius:18px; color:#101315; background:var(--lime); font-size:3.8rem; font-weight:900; line-height:1; }
    .answer-caption { margin:0 0 .3rem; color:var(--muted); font-size:.72rem; font-weight:800; letter-spacing:.13em; text-transform:uppercase; }
    .answer-title { margin:0; color:var(--ink); font-size:1.45rem; font-weight:800; }
    .answer-confidence { margin:.3rem 0 0; color:var(--lime); font-size:.9rem; font-weight:700; }
    .chart-heading { margin:1.3rem 0 .2rem; color:var(--ink); font-size:.92rem; font-weight:800; }
    .chart-subtitle { margin:0 0 .5rem; color:var(--muted); font-size:.78rem; }
    [data-testid="stMetric"] { background:#252e30; border:1px solid #34413f; border-radius:10px; padding:.6rem .7rem; }
    [data-testid="stMetricLabel"] { color:var(--muted) !important; }
    [data-testid="stMetricValue"] { color:var(--ink) !important; font-size:1.1rem !important; }
    .footer { padding-top:1.7rem; color:#65736f; font-size:.75rem; text-align:center; }
    </style>
    <div class="nav"><div class="logo"><span class="logo-mark">✦</span> Ink Lab</div><div class="nav-meta">MNIST / Neural vision</div></div>
    <div class="intro"><div><p class="eyebrow">Handwritten digit recognition</p><h1>Make a mark.<br>See what it means.</h1></div><p class="intro-copy">An interactive look at how a neural network interprets handwritten numbers.</p></div>
    """,
    unsafe_allow_html=True,
)

input_column, result_column = st.columns([1, 1], gap="large")
image: Image.Image | None = None

with input_column:
    st.markdown('<div class="workspace"><div class="workspace-label"><span>01 / Input</span><span>Live canvas</span></div>', unsafe_allow_html=True)
    st.markdown('<p class="input-help">Draw one large digit in the frame, or bring in an existing image.</p>', unsafe_allow_html=True)
    input_method = st.radio("Input method", ("Draw a digit", "Upload an image"), horizontal=True, label_visibility="collapsed")
    if input_method == "Upload an image":
        uploaded_file = st.file_uploader("Choose a digit image", type=("png", "jpg", "jpeg", "bmp"))
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Selected image", width=220)
    else:
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)", stroke_width=14, stroke_color="#101315",
            background_color="#f0f4ee", height=280, width=280, drawing_mode="freedraw", key="digit_canvas",
        )
        if canvas_result.image_data is not None:
            image = Image.fromarray(canvas_result.image_data.astype("uint8"), mode="RGBA")
    st.markdown('</div>', unsafe_allow_html=True)

with result_column:
    st.markdown('<div class="insight"><p class="insight-label">02 / Model output</p>', unsafe_allow_html=True)
    if image is None:
        st.markdown('<div class="waiting"><div class="waiting-symbol">✦</div><h3>Ready when you are</h3><p>Your prediction and probability profile will appear here after you draw or upload a digit.</p></div>', unsafe_allow_html=True)
    else:
        try:
            with st.spinner("Reading the mark..."):
                model = get_model()
                model_input = prepare_image(image)
            if not model_input.any():
                st.markdown('<div class="waiting"><div class="waiting-symbol">?</div><h3>Nothing to read yet</h3><p>Try a darker, larger mark with a clear background.</p></div>', unsafe_allow_html=True)
            else:
                probabilities = model.predict(model_input, verbose=0)[0]
                predicted_digit = int(probabilities.argmax())
                confidence = float(probabilities[predicted_digit])
                st.markdown(f'<div class="answer"><div class="answer-digit">{predicted_digit}</div><div><p class="answer-caption">Most likely digit</p><p class="answer-title">Digit {predicted_digit}</p><p class="answer-confidence">{confidence:.1%} confidence</p></div></div>', unsafe_allow_html=True)
                top_digits = probabilities.argsort()[-3:][::-1]
                st.markdown('<p class="chart-heading">Probability profile</p><p class="chart-subtitle">How strongly the model considered each class.</p>', unsafe_allow_html=True)
                chart_data = pd.DataFrame({"Confidence": probabilities}, index=[str(digit) for digit in range(10)])
                st.bar_chart(chart_data, color="#ff9b62", height=255)
                metric_columns = st.columns(3)
                for column, digit in zip(metric_columns, top_digits):
                    column.metric(f"Digit {int(digit)}", f"{probabilities[digit]:.1%}")
        except FileNotFoundError as error:
            st.error(str(error))
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="footer">Best results come from one centered digit with strong contrast.</div>', unsafe_allow_html=True)

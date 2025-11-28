import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model
import time
import os

# -----------------------------------------------------
# CSS Styling
# -----------------------------------------------------
st.markdown("""
    <style>

    /* Center the main content */
    .main {
        padding-top: 20px;
        padding-left: 10%;
        padding-right: 10%;
    }

    /* Card style containers */
    .card {
        padding: 25px;
        border-radius: 12px;
        background-color: #ffffff10;
        backdrop-filter: blur(6px);
        border: 1px solid rgba(255,255,255,0.15);
        box-shadow: 0px 4px 15px rgba(0,0,0,0.15);
        margin-bottom: 20px;
    }

    /* Title style */
    .title {
        text-align: center;
        font-size: 36px;
        font-weight: 800;
        margin-bottom: 15px;
        color: #4CAF50;
    }

    /* Upload text center */
    .upload-text {
        font-size: 25px;
        margin-top: 25px;
        font-weight: bold;
        margin-bottom: 8px;
    }
    
    </style>
""", unsafe_allow_html=True)

print("Current working directory:", os.getcwd())

# -----------------------------------------------------
# Load trained model
# -----------------------------------------------------
@st.cache_resource
def load_cache_model():
    with st.spinner("Loading model..."):
        time.sleep(1)
        try:
            model = load_model("models/mobilenet_bird_drone_classifier.keras")
        except FileNotFoundError:
            st.error("Model file not found: models/mobilenet_bird_drone_classifier.keras")
            model = None
    return model


model = load_cache_model()

# -----------------------------------------------------
# App Title
# -----------------------------------------------------
st.markdown('<div class="title">Aerial Object Classifier (Bird vs Drone)</div>', unsafe_allow_html=True)


st.markdown('<br />', unsafe_allow_html=True)

# -----------------------------------------------------
# Image Upload Section
# -----------------------------------------------------

st.markdown('<p class="upload-text">Upload an image</p>', unsafe_allow_html=True)
uploaded = st.file_uploader("", type=["jpg", "png"])

# -----------------------------------------------------
# When Image Uploaded
# -----------------------------------------------------
if uploaded:
    st.info("Processing uploaded image...")

    # Fake progress bar for upload UI
    progress = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress.progress(i + 1)

    img = Image.open(uploaded).resize((224, 224))
    st.image(img, caption="Uploaded Image", use_container_width=True)

    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)


    # -----------------------------------------------------
    # Prediction Section
    # -----------------------------------------------------

    st.write("Running prediction...")

    with st.spinner("Analyzing..."):
        time.sleep(0.5)
        pred = model.predict(arr)[0][0]

    # Confidence calculation
    bird_conf = float(pred) * 100
    drone_conf = float(100 - bird_conf)

    # Determine class
    if pred > 0.5:
        label = "Bird"
        conf = bird_conf
    else:
        label = "Drone"
        conf = drone_conf

    # Display result
    st.success(f"**Prediction:** {label}")

    # -----------------------------------------------------
    # Confidence Score Visualization
    # -----------------------------------------------------
    st.write(f"### Confidence: **{conf:.2f}%**")

    st.progress(int(conf))

    # Side-by-side probabilities
    st.write("### Probability Breakdown")
    st.write(f"**Bird:** {bird_conf:.2f}%")
    st.write(f"**Drone:** {drone_conf:.2f}%")

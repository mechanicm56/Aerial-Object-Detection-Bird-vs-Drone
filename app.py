import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model
import os

print("Current working directory:", os.getcwd())

# -------------------------------
# Load trained model
# -------------------------------
@st.cache_resource
def load_cache_model():
    try:
        model = load_model("models/mobilenet_bird_drone_classifier.keras")
    except FileNotFoundError:
        model = None
    return model

model = load_cache_model()

st.title("Aerial Object Classifier (Bird vs Drone)")

uploaded = st.file_uploader("Upload image", type=["jpg", "png"])

if uploaded:
    img = Image.open(uploaded).resize((224,224))
    st.image(img)

    arr = np.array(img)/255.0
    arr = np.expand_dims(arr, axis=0)

    pred = model.predict(arr)[0][0]

    if pred > 0.5:
        st.success("Prediction: Bird 🐦")
    else:
        st.success("Prediction: Drone 🛸")

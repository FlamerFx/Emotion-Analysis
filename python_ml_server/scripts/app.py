import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
from streamlit_lottie import st_lottie
import json
import os

# Load Lottie animation from file
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# Set up base directory and asset paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "python_ml_server", "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.keras")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
LOTTIE_PATH = os.path.join(MODEL_DIR, "Hello Animation.json")
IMAGE_PATH = os.path.join(MODEL_DIR, "Herta.png")

# Load model and assets only once
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    with open(LABEL_ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)
    return model, tokenizer, label_encoder

model, tokenizer, label_encoder = load_assets()
MAX_LEN = 100

# Welcome screen
if "page" not in st.session_state:
    st.session_state.page = "welcome"

if st.session_state.page == "welcome":
    lottie_hello = load_lottiefile(LOTTIE_PATH)
    st_lottie(lottie_hello, speed=1, width=300, height=300, key="hello")
    st.image(IMAGE_PATH, width=200)
    st.markdown("""
        <h2 style='text-align: center;'>Hello! My name is <span style='color:#8e44ad'>Herta</span> 🌀</h2>
        <p style='text-align: center;'>I was created by <b>Sitanshu & Yash</b> for a semester project.<br>
        I can predict your emotions through text!<br>
        <i>I'm a semester project, so I can make mistakes.</i></p>
    """, unsafe_allow_html=True)
    if st.button("Continue"):
        st.session_state.page = "classifier"
    st.stop()

# Emotion classifier UI
st.title("Emotion Classifier 🤖")
st.write("Enter your text below and I'll try to guess your emotion:")

user_input = st.text_input("Your text:")

if st.button("Predict"):
    try:
        seq = tokenizer.texts_to_sequences([user_input])
        padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=MAX_LEN, padding="post")
        pred = model.predict(padded)
        emotion = label_encoder.inverse_transform([np.argmax(pred)])
        st.success(f"**Predicted emotion:** {emotion[0]}")
    except Exception as e:
        st.error(f"Prediction error: {e}")
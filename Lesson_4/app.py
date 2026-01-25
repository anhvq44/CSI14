import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image

import numpy as np
from PIL import Image, ImageOps

st.set_page_config(
    page_title="ASL Alphabet Recognition",
    page_icon="🤘",
    layout="centered",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model():
    model = keras.models.load_model("asl_alphabet_model.h5")
    return model

model = load_model()

class_names = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
    'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'space',
    'nothing'
]
IMG_SIZE = 64

def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image_array = np.array(image)
    image_array = image_array / 255.0  # normalize to [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # add batch dimension
    return image_array

st.title("ASL Alphabet Recognition")

input_type = st.radio(
    "Choose input type:",
    ("Upload Image", "Use Kebcam"),
    index=0
)

uploaded_file = st.file_uploader("Upload an image of a hand sign:",
                                 type=["jpg", "jpeg", "png"])

if input_type == "Upload Image" and uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded image", use_container_width=True)
    
    # du doan % thap nhat cho phep
    THRESHOLD = 0.7
    
    if st.button("Predict"):
        with st.spinner("Predicting..."):
            img = preprocess_image(img)
            predictions = model.predict(img)
            print(predictions)
            confidence = np.max(predictions)
            if confidence < THRESHOLD:
                predicted_class = "Uncertain Prediction"
            else:
                predicted_class = class_names[np.argmax(predictions)]
            
        st.success(f"Predictions: {predicted_class} (Confidence: {confidence:.2f})")
        
        
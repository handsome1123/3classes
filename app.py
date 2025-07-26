import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model('banknote_model_mobilenet.h5')

# Define your class names (adjust if different)
class_names = ['1000', '5000', '10000']

# App title
st.title("💵 Banknote Denomination Classifier")

# Image upload
uploaded_file = st.file_uploader("Upload a banknote image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    pred = model.predict(img_array)
    pred_class = np.argmax(pred, axis=1)[0]
    confidence = np.max(pred)

    st.markdown(f"### 🧠 Prediction: **{class_names[pred_class]} MMK**")
    st.markdown(f"Confidence: **{confidence*100:.2f}%**")

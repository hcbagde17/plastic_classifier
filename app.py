import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

CLASS_NAMES = ['Mix HD', 'Mix PET','Mix PP','Mix Rigid']

@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "Plastic_type_classifier.keras")
    return tf.keras.models.load_model(model_path)

model = load_model()

st.title("Plastic Type Classifier")
st.write("Upload an image with platic items:")

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    #Load image as RGB
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    
    image = image.resize((224, 224))
    image = np.array(image).astype(np.float32)

    # Add batch dimension
    image = np.expand_dims(image, axis=0)  # (1,224,224,3)

    #Predict
    prediction = model.predict(image)
    predicted_index = int(np.argmax(prediction))
    predicted_food = CLASS_NAMES[predicted_index]

    st.subheader(f"Predicted Plastic: **{predicted_food}**")

    # Show probability distribution
    st.bar_chart(prediction[0])

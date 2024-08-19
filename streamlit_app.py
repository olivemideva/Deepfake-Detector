import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
from PIL import Image

# Load the pre-trained model
model = load_model('model/cnn_deepfake_model.keras')

# Function to preprocess the uploaded image
def preprocess_image(image):
    image_size = (64, 64)
    img = np.array(image)
    img = cv2.resize(img, image_size)
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  
    return img

# Function to make predictions
def make_prediction(img):
    prediction = model.predict(img)
    if prediction[0][1] > 0.5:
        return "Fake"
    else:
        return "Real"

# Streamlit app
st.title("Deepfake Detection")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=False, width=200)

    # Preprocess and predict
    img = preprocess_image(image)
    prediction = make_prediction(img)

    # Display prediction
    st.write(f"Prediction: **{prediction}**")

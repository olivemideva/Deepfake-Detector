import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
model = load_model('model/cnn_deepfake_model.keras')

# Set the image size to match the preprocessing step
image_size = (64, 64)

# Function to preprocess the uploaded image
def preprocess_image(image):
    img = cv2.resize(np.array(image), image_size)
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Streamlit app
st.title("Deepfake Detection")
st.write("Upload an image to see if it's real or fake.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)

    # Display the uploaded image in a small, consistent size
    st.image(image, caption='Uploaded Image', use_column_width=True, width=200)

    # Preprocess the image
    img = preprocess_image(image)

    # Make a prediction
    prediction = model.predict(img)

    # Determine if the image is real or fake
    if prediction[0][1] > 0.5:
        st.write("Prediction: Fake")
    else:
        st.write("Prediction: Real")

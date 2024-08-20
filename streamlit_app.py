import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
from PIL import Image

# Load the model
model = load_model('model/cnn_deepfake_model.keras')

# Function to preprocess the image
def preprocess_image(image):
    image_size = (64, 64)
    img = image.resize(image_size)  # Resize to match the model's expected input
    img = np.array(img, dtype=np.float32) / 255.0  # Normalize to 0-1
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Streamlit app
st.title("Deepfake Detection")
st.write("Upload an image to predict whether it is real or fake.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image in a smaller, consistent size
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True, width=150)

    # Convert the file to an image
    image = Image.open(uploaded_file)

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Predict
    prediction = model.predict(preprocessed_image)
    predicted_label = np.argmax(prediction, axis=1)[0]

    # Map prediction to label
    label_mapping = {0: 'REAL', 1: 'FAKE'}
    result = label_mapping[predicted_label]

    # Display the prediction
    st.write(f"Prediction: **{result}**")

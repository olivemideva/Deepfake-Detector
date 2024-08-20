import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
try:
    model = load_model('model/cnn_deepfake_model.keras')
    st.write("Model loaded successfully.")
except Exception as e:
    st.write("Error loading model:", str(e))

# Set the image size to match the preprocessing step
image_size = (64, 64)

# Function to preprocess the uploaded image
def preprocess_image(image):
    # Convert to numpy array
    img_array = np.array(image)
    
    # Resize the image to 64x64
    img_resized = cv2.resize(img_array, image_size)
    
    # Normalize pixel values to the range 0-1
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Ensure image has 3 channels (RGB)
    if img_normalized.ndim == 2:
        img_normalized = np.stack([img_normalized] * 3, axis=-1)
    
    # Add a batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch

# Streamlit app
st.title("Deepfake Detection")
st.write("Upload an image to see if it's real or fake.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file).convert('RGB')

    # Display the uploaded image in a small, consistent size
    st.image(image, caption='Uploaded Image', width=200)

    # Preprocess the image
    img = preprocess_image(image)

    # Make a prediction
    prediction = model.predict(img)
    
    # Print raw prediction for debugging
    st.write("Raw Prediction:", prediction)

    # Determine if the image is real or fake
    if prediction[0][1] > 0.5:
        st.write("Prediction: Fake")
    else:
        st.write("Prediction: Real")

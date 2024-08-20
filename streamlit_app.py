import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from io import BytesIO

# Load the trained CNN model
model = load_model('model/cnn_deepfake_model.keras')

def preprocess_image(image):
    """Preprocess the uploaded image to the format required by the model."""
    image_size = (64, 64)
    img = Image.open(image)
    img = img.resize(image_size)
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def predict_image(img):
    """Make a prediction on the preprocessed image."""
    prediction = model.predict(img)
    predicted_label = np.argmax(prediction, axis=1)[0]  # Get the predicted class (0 or 1)
    label_mapping = {0: 'REAL', 1: 'FAKE'}
    return label_mapping[predicted_label], prediction[0]

def display_image_with_prediction(image, prediction_label, prediction_prob):
    """Display an image with its prediction and confidence."""
    st.image(image, caption=f'Prediction: {prediction_label} (Confidence: {prediction_prob:.2f})', use_column_width=True)

def main():
    st.title("Deepfake Detection")
    st.write("Upload an image to check if it's REAL or FAKE.")

    # Image upload
    uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        st.write("Uploaded Images and Predictions:")
        
        # Iterate over uploaded files
        for uploaded_file in uploaded_files:
            # Display the uploaded image
            img = Image.open(uploaded_file)
            st.image(img, caption='Uploaded Image', use_column_width=True)

            # Preprocess the uploaded image
            processed_img = preprocess_image(uploaded_file)
            
            # Make prediction
            prediction_label, prediction_prob = predict_image(processed_img)
            
            # Display the prediction
            st.write(f"Prediction: **{prediction_label}**")
            st.write(f"Prediction Confidence: {prediction_prob[np.argmax(prediction_prob)]:.2f}")

            # Display image with prediction
            display_image_with_prediction(img, prediction_label, prediction_prob[np.argmax(prediction_prob)])

if __name__ == "__main__":
    main()

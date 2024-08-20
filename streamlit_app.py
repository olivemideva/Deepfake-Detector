import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained CNN model
model = load_model('model/cnn_deepfake_model.keras')

def preprocess_image(image):
    """Preprocess the uploaded image to the format required by the model."""
    image_size = (64, 64)  # Ensure this matches the input size of your model
    img = Image.open(image)
    img = img.convert('RGB')  # Convert to RGB to ensure compatibility
    img = img.resize(image_size)
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def predict_image(img):
    """Make a prediction on the preprocessed image."""
    prediction = model.predict(img)
    predicted_label = np.argmax(prediction, axis=1)[0]  # Get the predicted class (0 or 1)
    label_mapping = {0: 'REAL', 1: 'FAKE'}
    confidence = np.max(prediction)  # Get the highest confidence score
    return label_mapping[predicted_label], confidence

def main():
    st.title("Deepfake Detection")
    st.write("Upload images to check if they are REAL or FAKE.")
    
    uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    if uploaded_files:
        st.write("Uploaded Images and Predictions:")
        
        for uploaded_file in uploaded_files:
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded Image", use_column_width=True)  # Display original image
            
            # Preprocess the uploaded image
            processed_img = preprocess_image(uploaded_file)
            
            # Make prediction
            prediction_label, confidence = predict_image(processed_img)
            
            # Display result
            st.write(f"Prediction: **{prediction_label}** (Confidence: {confidence:.2f})")
            st.image(img.resize((200, 200)), caption=f'Prediction: {prediction_label}', use_column_width=False)  # Smaller image display

if __name__ == "__main__":
    main()

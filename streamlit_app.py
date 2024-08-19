import streamlit as st
from PIL import Image
import numpy as np
import cv2
from keras.models import load_model

# Load the model once at startup
model = load_model('model/cnn_model.h5')

def preprocess_image(image):
    image_size = (64, 64)  # Ensure consistency with your model's expected input size
    img = cv2.resize(image, image_size)  # Resize to match input size of model
    img = np.array(img, dtype=np.float32) / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def make_prediction(image):
    img = preprocess_image(image)
    prediction = model.predict(img)
    if prediction[0][1] > 0.5:
        return "Fake"
    else:
        return "Real"

def main():
    st.title("Deepfake Detection")
    st.write("Upload a photo to check if it's real or fake.")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert the image to OpenCV format
        image_cv = np.array(image)
        if image_cv.shape[2] == 4:  # Check if the image has an alpha channel
            image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGBA2RGB)  # Convert RGBA to RGB

        # Make prediction
        prediction = make_prediction(image_cv)
        st.write(f"Prediction: {prediction}")

if __name__ == "__main__":
    main()

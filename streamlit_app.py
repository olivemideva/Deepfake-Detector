import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
from PIL import Image

# Load the model
@st.cache(allow_output_mutation=True)
def load_prediction_model():
    model = load_model('model/cnn_deepfake_model.keras')
    return model

model = load_prediction_model()

# Preprocess the image
def preprocess_image(image):
    image_size = (64, 64)
    img = cv2.resize(np.array(image), image_size)
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Make prediction
def make_prediction(image):
    img = preprocess_image(image)
    prediction = model.predict(img)
    if prediction[0][1] > 0.5:
        return "Fake"
    else:
        return "Real"

# Streamlit app
st.title('Deepfake Detection')
st.write('Upload a photo to check if it\'s real or fake.')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    prediction = make_prediction(image)
    st.write(f'Prediction: {prediction}')

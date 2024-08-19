import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
from PIL import Image
import base64

# Load the model
@st.cache_resource(allow_output_mutation=True)
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

# Convert HTML, CSS, and JavaScript to Streamlit format
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def load_image(image_file):
    img = Image.open(image_file)
    return img

# Background image as base64
def get_img_as_base64(file_name):
    with open(file_name, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Apply HTML and CSS
def apply_html_css():
    st.markdown("""
        <style>
        body {
            font-family: 'Roboto', sans-serif;
            margin: 0;
            padding: 0;
            background-image: url('data:image/png;base64,{}');
            background-size: cover; 
            background-position: center;
            background-repeat: no-repeat;
        }
        .background-container {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            color: white;
        }
        .navbar {
            background: rgba(0, 0, 0, 0.4); 
            border-bottom: 1px solid rgba(255, 255, 255, 0.1); 
            width: 100%;
            padding: 10px 20px;
            position: fixed;
            top: 0;
            z-index: 9999;
        }
        .navbar-brand {
            font-family: 'Merriweather', serif;
            font-size: 1.75rem;
            color: white;
        }
        .container {
            max-width: 700px;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            margin-top: 5rem;
            background: rgba(255, 255, 255, 0.8);
        }
        h1 {
            font-family: 'Merriweather', serif;
            font-size: 2.5rem;
            margin-bottom: 1rem;
            color: #333;
        }
        .lead {
            color: #333;
        }
        .form-control {
            font-size: 1rem;
            margin-bottom: 1rem;
            border-radius: 8px;
            box-shadow: inset 0 1px 2px rgba(0,0,0,0.1);
        }
        .btn-primary {
            background-color: #343a40;
            border-color: rgba(255, 255, 255, 0.1);
            font-size: 1rem;
            padding: 0.5rem 1rem;
            border-radius: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        }
        footer {
            padding: 1rem;
            background-color: rgba(0, 0, 0, 0.4);
            color: #ffffff;
            text-align: center;
            position: fixed;
            bottom: 0;
            width: 100%;
        }
        footer p {
            margin: 0;
            font-size: 0.875rem;
        }
        .result {
            margin-top: 1rem;
            font-size: 1.25rem;
            color: #fff;
            background-color: rgba(0, 0, 0, 0.6);
            padding: 1rem;
            border-radius: 30px;
            display: none;
        }
        #image-preview {
            margin-top: 20px;
            text-align: center;
        }
        #uploaded-image {
            width: 300px;
            height: 300px;
            object-fit: cover;
            border-radius: 12px;
        }
        </style>
    """.format(get_img_as_base64("static/banner.jpg")), unsafe_allow_html=True)

apply_html_css()

# Streamlit app
st.markdown('<nav class="navbar navbar-expand-md navbar-dark fixed-top"><a class="navbar-brand" href="#">Deepfake Detection</a></nav>', unsafe_allow_html=True)

st.markdown('<div class="container"><h1>Deepfake Detection</h1><p class="lead">Upload a photo to check if it\'s real or fake.</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="file")

if uploaded_file is not None:
    image = load_image(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=False, output_format='auto', width=300)
    
    prediction = make_prediction(image)
    st.markdown(f'<div id="result" class="result" style="display:block;">Prediction: {prediction}</div>', unsafe_allow_html=True)

st.markdown('</div><footer><p>&copy; 2024 Deepfake Detection</p></footer>', unsafe_allow_html=True)

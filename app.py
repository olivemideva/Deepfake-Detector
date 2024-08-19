import streamlit as st
from PIL import Image
import numpy as np
from keras.models import load_model

# Load the model
model = load_model('model/cnn_deepfake_model.keras')

# Function to preprocess the image using PIL
def preprocess_image(image):
    image_size = (64, 64)
    img = image.resize(image_size)
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Streamlit application
def main():
    st.set_page_config(page_title="Deepfake Detection", layout="wide")
    
    # Background styling
    st.markdown("""
        <style>
            .reportview-container {
                background-image: linear-gradient(rgba(0,0,0,0.4), rgba(0,0,0,0.4)), url('static/banner.jpg');
                background-size: cover; 
                background-position: center;
                background-repeat: no-repeat; 
                height: 100vh; 
                width: 100vw;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                color: white;
                font-family: 'Roboto', sans-serif;
            }
            .stFileUploader {
                width: 100%;
            }
            .stButton {
                background-color: #343a40;
                color: white;
                border-radius: 30px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.2);
                padding: 0.5rem 1rem;
                font-size: 1rem;
            }
            .stText {
                color: #fff;
                background-color: rgba(0, 0, 0, 0.6);
                padding: 1rem;
                border-radius: 30px;
                font-size: 1.25rem;
            }
            .stImage {
                border-radius: 12px;
            }
        </style>
    """, unsafe_allow_html=True)
    
    st.title('Deepfake Detection')
    st.write('Upload a photo to check if it\'s real or fake.')

    # Upload an image file
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        
        # Preprocess the image and make prediction
        img = preprocess_image(image)
        prediction = model.predict(img)
        result = "Fake" if prediction[0][1] > 0.5 else "Real"
        
        st.markdown(f'<div class="stText">Prediction: {result}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()

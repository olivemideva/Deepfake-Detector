import streamlit as st
import numpy as np
from PIL import Image
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

# Streamlit app
st.title('Deepfake Detection')
st.write("Upload a photo to check if it's real or fake.")

uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess and make prediction
    img = preprocess_image(image)
    prediction = model.predict(img)
    
    # Display result
    if prediction[0][1] > 0.5:
        st.write('Prediction: Fake')
    else:
        st.write('Prediction: Real')

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import numpy as np
import cv2
from keras.models import load_model
import logging

# Initialize Flask app
app = Flask(__name__, template_folder='public/templates', static_folder='static')

# Enable CORS
CORS(app)

# Initialize model as None and load it lazily
model = None

def load_prediction_model():
    global model
    if model is None:
        app.logger.info("Loading deepfake detection model...")
        model = load_model('model/cnn_deepfake_model.keras')
    return model

# Function to preprocess the image using OpenCV
def preprocess_image(image_path):
    image_size = (64, 64)
    img = cv2.imread(image_path)
    img = cv2.resize(img, image_size)
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  
    return img

@app.route('/')
def index():
    return render_template('index.html')

# Configurations for file upload
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Log the file path
            app.logger.info(f"File saved at {file_path}")
            
            # Make prediction
            prediction = make_prediction(file_path)
            return jsonify({'prediction': prediction})
        else:
            return jsonify({'error': 'Invalid file format'}), 400
    except Exception as e:
        app.logger.error(f'Error occurred in /predict route: {e}')
        return jsonify({'error': 'Internal server error'}), 500

def make_prediction(file_path):
    # Load model lazily
    model = load_prediction_model()
    
    # Preprocess the image and make a prediction
    img = preprocess_image(file_path)
    prediction = model.predict(img)
    app.logger.info(f"Prediction result: {prediction}")
    
    if prediction[0][1] > 0.5:
        return "Fake"
    else:
        return "Real"

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(level=logging.DEBUG)

    # Ensure upload directory exists
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    # Run Flask app directly
    app.run(host='0.0.0.0', port=3000, debug=True)

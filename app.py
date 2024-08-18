from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import numpy as np
import cv2
import tensorflow as tf
import logging

app = Flask(__name__, template_folder='public/templates', static_folder='static')
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model
try:
    model = tf.keras.models.load_model('model/cnn_deepfake_model.keras', compile=False)
    logger.info('Model loaded successfully.')
except Exception as e:
    logger.error(f'Error loading model: {str(e)}')

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

def preprocess_image(image_path):
    image_size = (64, 64)
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Load image directly as color
    if img is None:
        raise ValueError("Image not found or invalid.")
    img = cv2.resize(img, image_size, interpolation=cv2.INTER_AREA)  # Use INTER_AREA for faster resizing
    img = img.astype(np.float32) / 255.0  # Normalizing
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        logger.error('No file part in the request')
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        logger.error('No selected file')
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        logger.info(f'File saved to {file_path}')
        try:
            prediction = make_prediction(file_path)
            logger.info(f'Prediction: {prediction}')
            return jsonify({'prediction': prediction})
        except Exception as e:
            logger.error(f'Error during prediction: {str(e)}')
            return jsonify({'error': 'Internal server error'}), 500
    else:
        logger.error('Invalid file format')
        return jsonify({'error': 'Invalid file format'}), 400

def make_prediction(file_path):
    try:
        img = preprocess_image(file_path)
        logger.info(f'Image shape after preprocessing: {img.shape}')
        prediction = model.predict(img, verbose=0)  # Disable verbose to speed up prediction
        logger.info(f'Model prediction: {prediction}')
        return "Fake" if prediction[0][1] > 0.5 else "Real"
    except Exception as e:
        logger.error(f'Error during image preprocessing or prediction: {str(e)}')
        raise

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def run_app():
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    # Get the port from the environment variable or default to 8080
    port = int(os.getenv("PORT", 8080))
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)  # Disable debug mode

if __name__ == '__main__':
    run_app()

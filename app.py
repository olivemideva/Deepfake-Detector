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
model_path = 'model/cnn_deepfake_model.keras'
if not os.path.exists(model_path):
    logger.error(f'Model file not found: {model_path}')
    raise FileNotFoundError(f'Model file not found: {model_path}')
model = tf.keras.models.load_model(model_path)
logger.info('Model loaded successfully.')

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

def preprocess_image(image_path):
    image_size = (64, 64)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found or invalid.")
    img = cv2.resize(img, image_size)
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
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
        try:
            prediction = make_prediction(file_path)
            logger.info(f'Prediction: {prediction}')
            return jsonify({'prediction': prediction})
        except Exception as e:
            logger.error(f'Error during prediction: {e}')
            return jsonify({'error': str(e)}), 500
    else:
        logger.error('Invalid file format')
        return jsonify({'error': 'Invalid file format'}), 400

def make_prediction(file_path):
    img = preprocess_image(file_path)
    prediction = model.predict(img)
    return "Fake" if prediction[0][1] > 0.5 else "Real"

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def run_app():
    port = int(os.environ.get('PORT', 8080))  # Use environment variable for port if available
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False)

if __name__ == '__main__':
    run_app()

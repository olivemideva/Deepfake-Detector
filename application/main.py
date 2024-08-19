from flask import Blueprint, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import numpy as np
import cv2

main = Blueprint('main', __name__)

def preprocess_image(image_path):
    image_size = (64, 64)
    img = cv2.imread(image_path)
    img = cv2.resize(img, image_size)
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  
    return img

@main.route('/')
def index():
    return render_template('index.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

@main.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        prediction = make_prediction(file_path)
        return jsonify({'prediction': prediction})
    else:
        return jsonify({'error': 'Invalid file format'}), 400

def make_prediction(file_path):
    img = preprocess_image(file_path)
    prediction = current_app.model.predict(img)
    
    if prediction[0][1] > 0.5:
        return "Fake"
    else:
        return "Real"

@main.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(current_app.config['UPLOAD_FOLDER'], filename)

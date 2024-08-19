from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
from keras.models import load_model
from keras import backend as K
import logging

# Initialize Flask app
app = Flask(__name__, template_folder='public/templates', static_folder='static')

# Load the model
model = load_model('model/cnn_deepfake_model.keras')

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Function to preprocess the image using PIL with added optimization
def preprocess_image(image_path):
    image_size = (64, 64)  # Resize images to the size expected by your model
    try:
        img = Image.open(image_path)
        img = img.convert('RGB')  # Ensure the image is in RGB format
        img = img.resize(image_size, Image.ANTIALIAS)  # Resize with antialiasing for better quality
        img = np.array(img, dtype=np.float32) / 255.0  # Normalize the image
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        logging.debug(f"Image processed successfully: {image_path}")
        return img
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return None

# Define allowed extensions
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        logging.error('No file part in the request')
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        logging.error('No file selected')
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save file temporarily for processing
        file.save(file_path)
        logging.debug(f"File saved: {file_path}")
        
        # Process the image and make a prediction
        img = preprocess_image(file_path)
        if img is None:
            logging.error('Error during image processing')
            return jsonify({'error': 'Error processing image'}), 500
        
        prediction = make_prediction(img)
        
        # Remove the uploaded file after processing to save space
        os.remove(file_path)
        logging.debug(f"File removed after processing: {file_path}")
        
        return jsonify({'prediction': prediction})
    else:
        logging.error('Invalid file format')
        return jsonify({'error': 'Invalid file format'}), 400

def make_prediction(img):
    try:
        prediction = model.predict(img)
        K.clear_session()  # Clear the session to free up resources
        if prediction[0][1] > 0.5:
            return "Fake"
        else:
            return "Real"
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return "Error"

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def run_app():
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True, use_reloader=False, threaded=True)

if __name__ == '__main__':
    # Running the app with Gunicorn is recommended for production
    # Use `gunicorn -w 4 -b 0.0.0.0:10000 app:app` for deployment
    app.run(host='0.0.0.0', port=10000, debug=True, use_reloader=False, threaded=True)

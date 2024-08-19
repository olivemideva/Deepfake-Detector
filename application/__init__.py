from flask import Flask
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import numpy as np
import cv2
from keras.models import load_model

def create_app():
    app = Flask(__name__, template_folder='public/templates', static_folder='static')
    
    # Enable CORS
    CORS(app)
    
    # Load the model
    app.model = load_model('model/cnn_deepfake_model.keras')
    
    # Configurations for file upload
    app.config['UPLOAD_FOLDER'] = 'uploads'
    app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
    
    # Register Blueprints
    from .main import main as main_blueprint
    app.register_blueprint(main_blueprint)

    # Ensure upload directory exists
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    return app

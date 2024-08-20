import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# Try to import scikit-learn; handle import error
try:
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    sklearn_available = True
except ImportError:
    sklearn_available = False

# Load the trained CNN model
model = load_model('model/cnn_deepfake_model.keras')

def preprocess_image(image):
    """Preprocess the uploaded image to the format required by the model."""
    image_size = (64, 64)  # Ensure this matches the input size of your model
    img = Image.open(image)
    img = img.convert('RGB')  # Convert to RGB to ensure compatibility
    img = img.resize(image_size)
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def predict_image(img):
    """Make a prediction on the preprocessed image."""
    prediction = model.predict(img)
    predicted_label = np.argmax(prediction, axis=1)[0]  # Get the predicted class (0 or 1)
    label_mapping = {0: 'REAL', 1: 'FAKE'}
    confidence = np.max(prediction)  # Get the highest confidence score
    return label_mapping[predicted_label], confidence

def evaluate_model():
    """Evaluate the model on a test dataset."""
    import os  # Import os locally to avoid global issues
    # Load test data
    real_test_path = 'dataset/test/REAL'
    fake_test_path = 'dataset/test/FAKE'

    def load_images_from_folder(folder, label):
        images = []
        for filename in os.listdir(folder):
            img_path = os.path.join(folder, filename)
            try:
                img = Image.open(img_path)
                img = img.resize((64, 64))
                img = np.array(img)
                images.append([img, label])
            except Exception as e:
                print(f"Error loading image {filename}: {e}")
        return images

    real_test_images = load_images_from_folder(real_test_path, label=0)  # Label 0 for real
    fake_test_images = load_images_from_folder(fake_test_path, label=1)  # Label 1 for fake
    all_test_images = real_test_images + fake_test_images
    df_test = pd.DataFrame(all_test_images, columns=['image', 'label'])

    X_test = np.array([img for img, _ in df_test['image']], dtype=np.float32)
    y_test = np.array([label for _, label in df_test['image']], dtype=np.float32)
    X_test = X_test / 255.0
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)

    # Predict on the test data
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Calculate accuracy
    accuracy = np.mean(y_pred_classes == y_true)

    if sklearn_available:
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred_classes)
        return accuracy, cm
    else:
        st.warning("scikit-learn is not available. Confusion Matrix cannot be displayed.")
        return accuracy, None

def main():
    st.title("Deepfake Detection")
    st.write("Upload images to check if they are REAL or FAKE.")
    
    uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    if uploaded_files:
        st.write("Uploaded Images and Predictions:")
        
        for uploaded_file in uploaded_files:
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded Image", use_column_width=True)  # Display original image
            
            # Preprocess the uploaded image
            processed_img = preprocess_image(uploaded_file)
            
            # Make prediction
            prediction_label, confidence = predict_image(processed_img)
            
            # Display result
            st.write(f"Prediction: **{prediction_label}** (Confidence: {confidence:.2f})")
            st.image(img.resize((200, 200)), caption=f'Prediction: {prediction_label}', use_column_width=False)  # Smaller image display
    
    # Evaluate model
    if st.button("Evaluate Model"):
        accuracy, cm = evaluate_model()
        st.write(f"Model Accuracy: **{accuracy:.2f}**")
        
        if cm is not None:
            # Display Confusion Matrix
            cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['REAL', 'FAKE'])
            st.write("Confusion Matrix:")
            st.pyplot(cm_display.plot(include_values=True, cmap='Blues').figure)

if __name__ == "__main__":
    main()

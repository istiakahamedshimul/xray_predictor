import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)

# Load the trained model once when the app starts
try:
    model = tf.keras.models.load_model('chest_xray_model.h5')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Define the prediction function
def predict_pneumonia(img_path):
    if model is None:
        return "Error: Model not loaded."

    # Load and preprocess the image
    img = Image.open(img_path).resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # The model was trained with images rescaled to 1/255.0
    img_array = img_array.astype('float32') / 255.0

    # Make a prediction
    prediction = model.predict(img_array)

    # Return the class based on the prediction
    if prediction[0][0] > 0.5:
        return "PNEUMONIA"
    else:
        return "NORMAL"

@app.route('/')
def index():
    """Renders the main page with the upload form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the image upload and returns a prediction."""
    if 'file' not in request.files:
        return "No file part in the request."
    file = request.files['file']
    if file.filename == '':
        return "No file selected."
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join("uploads", filename)
        
        # Create an 'uploads' directory if it doesn't exist
        os.makedirs("uploads", exist_ok=True)
        file.save(filepath)

        # Get the prediction
        prediction_result = predict_pneumonia(filepath)

        # Clean up the uploaded file
        os.remove(filepath)
        
        return render_template('index.html', prediction=prediction_result)

if __name__ == '__main__':
    app.run(debug=True)
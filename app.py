import os
import requests
import numpy as np
import tensorflow as tf
from flask import Flask, render_template_string, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image

# Ensure TensorFlow uses the CPU, as free Render instances do not have a GPU.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Use a placeholder URL for the model. You'll need to replace this
# with the actual URL where you host your model file.
MODEL_URL = "https://example.com/chest_xray_model.h5"
MODEL_PATH = 'chest_xray_model.h5'
TEMP_UPLOAD_DIR = "/tmp"

# Create the temp directory if it doesn't exist
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

# Function to download the model file
def download_model(url, path):
    """
    Downloads a file from a URL to a specified path.
    """
    print("Model not found locally. Attempting to download...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        with open(path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Model downloaded successfully!")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading model: {e}")
        return False

# Load the trained model. This is the key change.
# We check if the model exists locally first.
if not os.path.exists(MODEL_PATH):
    if not download_model(MODEL_URL, MODEL_PATH):
        # Handle the case where the download fails
        model = None
        print("Model not available. The application will not function correctly.")
    else:
        # Load the model after a successful download
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully!")
else:
    # If the model already exists (e.g., from a previous run or git repo), load it.
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully from local file!")

app = Flask(__name__)

# Define the prediction function
def predict_pneumonia(img_path):
    if model is None:
        return {"error": "Model not loaded."}

    try:
        # Load and preprocess the image
        img = Image.open(img_path).resize((224, 224)).convert('RGB')
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array.astype('float32') / 255.0

        # Make a prediction
        prediction = model.predict(img_array)

        # Return the class and probability
        result = "PNEUMONIA" if prediction[0][0] > 0.5 else "NORMAL"
        probability = float(prediction[0][0])
        
        return {"result": result, "probability": probability}
    except Exception as e:
        return {"error": str(e)}

@app.route('/', methods=['GET'])
def index():
    # Render the HTML using Flask's render_template_string
    return render_template_string('''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Pneumonia Predictor</title>
            <script src="https://cdn.tailwindcss.com"></script>
            <style>
                body {
                    font-family: 'Inter', sans-serif;
                }
                .process-step.completed {
                    color: green;
                    font-weight: bold;
                }
            </style>
        </head>
        <body class="bg-gray-100 flex items-center justify-center min-h-screen">
            <div class="bg-white p-8 rounded-xl shadow-lg w-full max-w-md">
                <h1 class="text-3xl font-bold text-center text-gray-800 mb-6">Pneumonia Predictor</h1>
                <p class="text-center text-gray-600 mb-8">Upload a chest X-ray image to get a prediction.</p>

                <form id="upload-form" action="/predict" method="post" enctype="multipart/form-data" class="space-y-4">
                    <label class="block">
                        <span class="sr-only">Choose file</span>
                        <input type="file" name="file" id="file-input" accept="image/*" class="block w-full text-sm text-gray-500
                            file:mr-4 file:py-2 file:px-4
                            file:rounded-full file:border-0
                            file:text-sm file:font-semibold
                            file:bg-blue-50 file:text-blue-700
                            hover:file:bg-blue-100
                        "/>
                    </label>
                    <button type="submit" id="submit-button" class="w-full py-2 px-4 bg-blue-600 text-white font-semibold rounded-full shadow-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition ease-in-out duration-150">
                        Predict
                    </button>
                </form>

                <div id="process-status" class="mt-8 text-gray-600 space-y-2 hidden">
                    <p class="text-lg font-bold">Prediction Process:</p>
                    <p id="step-upload" class="process-step">1. Uploading image...</p>
                    <p id="step-preprocess" class="process-step">2. Preprocessing image...</p>
                    <p id="step-predict" class="process-step">3. Making a prediction...</p>
                </div>

                <div id="prediction-result" class="mt-8 p-4 rounded-xl text-center hidden">
                    <!-- Result will be inserted here by JavaScript -->
                </div>
            </div>

            <script>
                document.getElementById('upload-form').addEventListener('submit', async function(event) {
                    event.preventDefault();
                    
                    const form = event.target;
                    const formData = new FormData(form);
                    const fileInput = document.getElementById('file-input');
                    
                    // Show processing status and clear previous results
                    const statusDiv = document.getElementById('process-status');
                    const resultDiv = document.getElementById('prediction-result');
                    statusDiv.classList.remove('hidden');
                    resultDiv.classList.add('hidden');
                    document.querySelectorAll('.process-step').forEach(el => el.classList.remove('completed'));
                    
                    const submitButton = document.getElementById('submit-button');
                    submitButton.disabled = true;
                    submitButton.textContent = 'Processing...';

                    // Simulate the first step immediately
                    document.getElementById('step-upload').classList.add('completed');
                    
                    try {
                        // Use a fetch request to handle the form submission
                        const response = await fetch(form.action, {
                            method: 'POST',
                            body: formData
                        });

                        // Step 2: Preprocessing
                        document.getElementById('step-preprocess').classList.add('completed');

                        // Step 3: Prediction
                        document.getElementById('step-predict').classList.add('completed');

                        const result = await response.json();

                        // Display the result
                        if (result.error) {
                            resultDiv.classList.remove('hidden');
                            resultDiv.className = 'mt-8 p-4 rounded-xl text-center bg-red-100 text-red-700';
                            resultDiv.innerHTML = `<p class="font-bold text-lg">Error:</p><p class="mt-2">${result.error}</p>`;
                        } else {
                            resultDiv.classList.remove('hidden');
                            const isPneumonia = result.result === 'PNEUMONIA';
                            resultDiv.className = `mt-8 p-4 rounded-xl text-center ${isPneumonia ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'}`;
                            resultDiv.innerHTML = `
                                <p class="font-bold text-lg">Prediction Result:</p>
                                <p class="text-2xl mt-2 font-bold">${result.result}</p>
                            `;
                        }

                    } catch (error) {
                        resultDiv.classList.remove('hidden');
                        resultDiv.className = 'mt-8 p-4 rounded-xl text-center bg-red-100 text-red-700';
                        resultDiv.innerHTML = `<p class="font-bold text-lg">An error occurred:</p><p class="mt-2">${error.message}</p>`;
                    } finally {
                        submitButton.disabled = false;
                        submitButton.textContent = 'Predict';
                    }
                });
            </script>
        </body>
        </html>
    ''')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request."})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected."})

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(TEMP_UPLOAD_DIR, filename)

        try:
            # Save the file to a temporary location
            file.save(filepath)

            # Get the prediction
            prediction_data = predict_pneumonia(filepath)
            return jsonify(prediction_data)
        except Exception as e:
            return jsonify({"error": str(e)})
        finally:
            # Clean up the uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)

if __name__ == '__main__':
    # You can still run the app directly for local testing
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 5000))

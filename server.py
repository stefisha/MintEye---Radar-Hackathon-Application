from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import json
import requests
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import pickle
import traceback
from dotenv import load_dotenv  # Import the dotenv package
import logging
from logging.handlers import RotatingFileHandler, WatchedFileHandler

# Load environment variables from .env file
load_dotenv()

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Flask app setup
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Configure logging
if not app.debug:
    if not os.path.exists('logs'):
        os.mkdir('logs')
    # Use WatchedFileHandler instead of RotatingFileHandler
    file_handler = WatchedFileHandler('logs/app.log')
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('App startup')

# Access environment variables from the .env file
RPC_URL_DAS = os.getenv('RPC_URL_DAS')

# Verify that the environment variables are loaded
if not RPC_URL_DAS:
    raise ValueError("Missing environment variables. Please check your .env file.")

# Load the MobileNetV3 fine-tuned image model
image_model = tf.keras.models.load_model("models/mobilenetv3_tunned.keras", compile=False)

# Load the pre-trained metadata models
metadata_models = []
for i in range(5):
    with open(f'models/model_{i}.pkl', 'rb') as f:
        metadata_models.append(pickle.load(f))

class_names = {
    0: 'legit',  # Legit (formerly regular)
    1: 'scam'    # Scam
}

# MobileNetV3 preprocessing function
def preprocess_mobnet_image(image):
    image = image.convert("RGB")  # Ensure 3 channels
    image = image.resize((256, 256))  # Resize to 256x256 as required by MobileNetV3
    image_array = np.array(image)
    image_array = tf.keras.applications.mobilenet_v3.preprocess_input(image_array)  # Preprocess
    return image_array

# Function to download an image from a URL
def download_image(url):
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()  # Check for request errors
        return Image.open(io.BytesIO(response.content))  # Load image in memory
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Can't download image from {url}: {e}")
        return None

# Function to extract the image URL from the metadata
def extract_image_url(metadata):
    if 'content' in metadata and 'files' in metadata['content']:
        files = metadata['content']['files']
        if isinstance(files, list) and len(files) > 0 and 'uri' in files[0]:
            return files[0]['uri']
    
    return metadata.get('json', {}).get('image') or metadata.get('image') or metadata.get('uri')

# Function to predict from image using MobileNetV3 model
def predict_from_image(image_model, image):
    preprocessed_image = preprocess_mobnet_image(image)
    image_batch = np.expand_dims(preprocessed_image, axis=0)
    probs = image_model.predict(image_batch)
    top_prob = float(probs.max())  # Ensure probability is a native Python float
    top_pred = class_names[probs.argmax()]
    return top_prob, top_pred

# Function to predict from metadata using XGBoost models
def predict_from_metadata(metadata, metadata_models):
    predictions = []
    for model, features in metadata_models:
        if all(feature in metadata for feature in features):
            input_data = [metadata[feature] for feature in features]
            pred = model.predict([input_data])[0]
            predictions.append(pred)
    
    if predictions:
        return max(set(predictions), key=predictions.count)  # Majority vote
    else:
        return None  # No prediction if no model could be used

# Function to handle the weighted average
def weighted_average(predictions, weights):
    weighted_sum = sum(p * w for p, w in zip(predictions, weights))
    total_weight = sum(weights)
    return weighted_sum / total_weight

# Function to fetch metadata using NFT mint address
def fetch_nft_metadata(mint_address):
    try:
        headers = {'Content-Type': 'application/json'}
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getAsset",
            "params": {"id": mint_address}
        }

        # Request metadata from RPC DAS
        response = requests.post(RPC_URL_DAS, headers=headers, json=payload)
        
        # Log the status code and response for debugging
        app.logger.info(f"Fetching metadata for mint: {mint_address}")
        app.logger.info(f"Status Code: {response.status_code}")
        app.logger.info(f"Response: {response.text}")

        data = response.json()

        if 'result' in data and data['result']:
            return data['result']
        else:
            app.logger.error(f"Could not fetch metadata for the NFT: {mint_address}")
            raise Exception("Could not fetch metadata for the given NFT.")
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Error fetching NFT metadata for {mint_address}: {e}")
        raise Exception(f"Error fetching NFT metadata: {str(e)}")

# Home page - renders the index.html
@app.route('/')
def index():
    return render_template('index.html')

# Handle mint address input and predict
@app.route('/fetch_nft', methods=['POST'])
def fetch_nft():
    try:
        if 'nft_mint' not in request.form or not request.form['nft_mint']:
            return jsonify({"error": "No mint address provided"}), 400

        # Fetch the mint address from the form
        mint_address = request.form['nft_mint']

        # Fetch metadata for the NFT using the mint address
        metadata = fetch_nft_metadata(mint_address)

        # Extract image URL and download image
        image_url = extract_image_url(metadata)
        image_prediction = None
        image_prob = None

        if image_url:
            image = download_image(image_url)
            if image:
                image_prob, image_pred = predict_from_image(image_model, image)
                image_prediction = 1 if image_pred == 'scam' else 0
            else:
                app.logger.warning(f"Image download failed, proceeding with metadata-only prediction.")
        
        # Predict from metadata
        metadata_prediction = predict_from_metadata(metadata, metadata_models)

        # Combine predictions
        final_prediction = None
        if image_prediction is not None and metadata_prediction is not None:
            final_prediction = weighted_average([image_prediction, metadata_prediction], weights=[0.5, 0.5])
        elif image_prediction is not None:
            final_prediction = image_prediction
        elif metadata_prediction is not None:
            final_prediction = metadata_prediction
        else:
            return jsonify({"error": "No valid prediction could be made"}), 500

        # Return the results
        prediction_label = "scam" if final_prediction >= 0.5 else "legit"
        probability_output = round(final_prediction, 2) if image_prob is None else round(image_prob, 2)

        app.logger.info(f"Prediction for {mint_address}: {prediction_label}, Probability: {probability_output}")

        return jsonify({
            "image_url": image_url if image_url else None,
            "prediction": prediction_label,
            "probability": probability_output
        })

    except Exception as e:
        traceback.print_exc()
        app.logger.error(f"Error in fetch_nft route: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

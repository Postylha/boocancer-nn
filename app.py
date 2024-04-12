from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from azure.storage.blob import BlobServiceClient
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import os

app = Flask(__name__)


# Function to download the model from Azure Blob Storage
def download_model_blob(storage_connection_string, container_name, blob_name, local_file_name):
    blob_service_client = BlobServiceClient.from_connection_string(storage_connection_string)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    os.makedirs(os.path.dirname(local_file_name), exist_ok=True)
    with open(local_file_name, "wb") as download_file:
        download_file.write(blob_client.download_blob().readall())


# Function to preprocess the image for your model
def preprocess_image(img, target_size=(224, 224)):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return img_array_expanded_dims


# Define the path for the local model
MODEL_LOCAL_PATH = '/tmp/model.h5'

# Download and load the Keras model if it's not already present
if not os.path.isfile(MODEL_LOCAL_PATH):
    # Set these in your environment variables or Azure App Service Configuration
    storage_connection_string = os.environ.get('AZURE_STORAGE_CONNECTION_STRING')
    container_name = 'cancerst'  # Replace with your actual container name
    blob_name = os.environ.get('MODEL_BLOB_NAME')

    if not storage_connection_string or not blob_name:
        raise ValueError("Azure Storage connection string and blob name must be set in environment variables")

    download_model_blob(storage_connection_string, container_name, blob_name, MODEL_LOCAL_PATH)

# Load the trained model
model = load_model(MODEL_LOCAL_PATH)


@app.route('/test', methods=['POST'])
def test():
    data = request.get_json()
    if data is None or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    # Decode the base64 image
    im_b64 = data['image']
    im_bytes = base64.b64decode(im_b64)
    im_file = BytesIO(im_bytes)
    img = Image.open(im_file)

    # Preprocess the image for the model
    processed_image = preprocess_image(img)

    # Make a prediction
    prediction = model.predict(processed_image)

    # Return the prediction in JSON format
    response_data = {'prediction': prediction.tolist()}  # Convert numpy array to list for JSON serialization

    return jsonify(response_data), 200


if __name__ == '__main__':
    app.run(debug=True, port=8080)

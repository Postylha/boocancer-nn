from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import os

app = Flask(__name__)

def preprocess_image(img, target_size=(224, 224)):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size)
    img_array = img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return img_array_expanded_dims


# Define the path for the local model
MODEL_LOCAL_PATH = '/mounted/model.h5'

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
    result = None
    prediction = model.predict(processed_image)
    if prediction > 0.5:
        result = 1
    else:
        result = 0
    
    # Return the prediction in JSON format
    response_data = {'prediction': result}  # Convert numpy array to list for JSON serialization

    return jsonify(response_data), 200

if __name__ == '__main__':
    app.run(debug=True, port=8080)

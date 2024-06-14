import os
import numpy as np
import keras
from keras import layers
import tensorflow as tf
from tensorflow import data as tf_data
from flask import Flask, request, jsonify
import matplotlib.pyplot as plt
import cv2
import imghdr
import json

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Input, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model



model = load_model('my_model.h5')

app = Flask(__name__)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Load the original class indices from the JSON file
with open('plantnet300K_species_id_2_name.json', 'r') as f:
    original_class_indices = json.load(f)

# Create a new mapping from 0 to len(original_class_indices) - 1
class_indices = {str(i): label for i, (original_idx, label) in enumerate(original_class_indices.items())}

# Save the new mapping to a new JSON file (optional)
with open('new_class_indices.json', 'w') as f:
    json.dump(class_indices, f)

    
    
def preprocess_input(image_batch):
    # Preprocess the input image here
    # This is an example, modify it according to your needs
    return tf.keras.applications.mobilenet_v2.preprocess_input(image_batch)

def custom_decode_predictions(preds, class_indices, top=5):
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [(class_indices[str(i)], float(pred[i])) for i in top_indices]
        results.append(result)
    return results

def process_images(model, image_paths, size, preprocess_input, top_k=2):
    results = []
    for idx, image_path in enumerate(image_paths):
        tf_image = tf.io.read_file(image_path)
        decoded_image = tf.image.decode_image(tf_image)
        image_resized = tf.image.resize(decoded_image, size)
        image_batch = tf.expand_dims(image_resized, axis=0)
        image_batch = preprocess_input(image_batch)
        preds = model.predict(image_batch)
        decoded_preds = custom_decode_predictions(preds, class_indices, top=top_k)
        results.append(decoded_preds)
    return results


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    image_paths = data['image_paths']
    size = (256, 256)
    top_k = data.get('top_k', 5)
    results = process_images(model, image_paths, size, preprocess_input, top_k)
    return jsonify(results)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))  # Use PORT environment variable or default to 5000
    app.run(debug=True, host='0.0.0.0', port=port)

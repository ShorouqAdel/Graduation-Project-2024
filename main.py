import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.layers import DepthwiseConv2D


# Initialize the FastAPI app
app = FastAPI()

# Load the model and class indices
model_path = 'my_model.h5'
class_indices_path = 'plantnet300K_species_id_2_name.json'

# Ensure GPU memory growth is enabled
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
        
custom_objects = {'DepthwiseConv2D': DepthwiseConv2D}
model = load_model(model_path, custom_objects=custom_objects)

# Load the original class indices from the JSON file
with open(class_indices_path, 'r') as f:
    original_class_indices = json.load(f)

# Create a new mapping from 0 to len(original_class_indices) - 1
class_indices = {str(i): label for i, (original_idx, label) in enumerate(original_class_indices.items())}

# Save the new mapping to a new JSON file (optional)
with open('new_class_indices.json', 'w') as f:
    json.dump(class_indices, f)

# Define request and response models
class PredictionRequest(BaseModel):
    image_paths: List[str]
    top_k: int = 5

class PredictionResponse(BaseModel):
    predictions: List[List[dict]]

def custom_decode_predictions(preds, class_indices, top=5):
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [{'label': class_indices[str(i)], 'probability': float(pred[i])} for i in top_indices]
        results.append(result)
    return results

def process_images(model, image_paths, size, preprocess_input, top_k=5):
    all_preds = []
    for image_path in image_paths:
        # Read the image using TensorFlow
        tf_image = tf.io.read_file(image_path)

        # Decode the above `tf_image` from a Bytes string to a numeric Tensor
        decoded_image = tf.image.decode_image(tf_image)

        # Resize the image to the spatial size required by the model
        image_resized = tf.image.resize(decoded_image, size)

        # Add a batch dimension to the first axis (required)
        image_batch = tf.expand_dims(image_resized, axis=0)

        # Pre-process the input image
        image_batch = preprocess_input(image_batch)

        # Forward pass through the model to make predictions
        preds = model.predict(image_batch)

        # Decode (and rank the top-k) predictions
        decoded_preds = custom_decode_predictions(preds, class_indices, top=top_k)
        all_preds.append(decoded_preds)

    return all_preds

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        size = (256, 256)
        predictions = process_images(model, request.image_paths, size, preprocess_input, request.top_k)
        return PredictionResponse(predictions=predictions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)

import os
import tensorflow as tf
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List
import json
import uvicorn

app = FastAPI()

# Set up GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Try to load the model
try:
    model = tf.keras.models.load_model('my_model.h5')
    print("Model loaded successfully")
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

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
    return tf.keras.applications.mobilenet_v2.preprocess_input(image_batch)

def custom_decode_predictions(preds, class_indices, top=5):
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [(class_indices[str(i)], float(pred[i])) for i in top_indices]
        results.append(result)
    return results

async def process_images(model, image_files, size, preprocess_input, top_k=2):
    results = []
    for image_file in image_files:
        try:
            contents = await image_file.read()
            tf_image = tf.image.decode_image(contents)
            image_resized = tf.image.resize(tf_image, size)
            image_batch = tf.expand_dims(image_resized, axis=0)
            image_batch = preprocess_input(image_batch)
            preds = model.predict(image_batch)
            decoded_preds = custom_decode_predictions(preds, class_indices, top=top_k)
            results.append(decoded_preds)
        except Exception as e:
            results.append(f"Error processing image {image_file.filename}: {e}")
    return results

@app.post('/predict', response_model=list)
async def predict(files: List[UploadFile] = File(...), top_k: int = 5):
    if model is None:
        raise HTTPException(status_code=500, detail="Model could not be loaded")

    if not files:
        raise HTTPException(status_code=400, detail="No image files provided")
    
    size = (256, 256)
    results = await process_images(model, files, size, preprocess_input, top_k)
    return results

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))  # Use PORT environment variable or default to 8000
    uvicorn.run(app, host='0.0.0.0', port=port)

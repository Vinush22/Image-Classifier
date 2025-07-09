# test.py

import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

def predict_image(model_path, img_path, img_size=(180, 180)):
    """
    Loads a saved model and predicts whether the given image is a cat or dog.
    """
    print("[INFO] Loading model...")
    model = tf.keras.models.load_model(model_path)

    print("[INFO] Loading image...")
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch

    print("[INFO] Predicting...")
    predictions = model.predict(img_array)
    score = tf.nn.sigmoid(predictions[0])

    print(f"\n📷 Prediction Result:")
    print(f"🦴 Dog Confidence: {100 * score:.2f}%")
    print(f"🐱 Cat Confidence: {100 * (1 - score):.2f}%")

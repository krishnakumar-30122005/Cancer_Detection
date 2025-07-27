import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

# Load models
classifier_model = load_model(r"D:/cancer detection/cancerDetection/models/classifier_model.h5")
brain_model = load_model(r"D:/cancer detection/cancerDetection/models/train_cnn_brain.h5")
skin_model = load_model(r"D:/cancer detection/cancerDetection/models/train_cnn_skin.h5")
breast_model = load_model(r"D:/cancer detection/cancerDetection/models/train_cnn_breast.h5")

# Label mappings
cancer_labels = ['Brain Tumor', 'Skin Cancer', 'Breast Cancer']
prediction_labels = ['cancer', 'normal']

# --- Preprocessing Functions ---

def preprocess_image(img_path, target_size):
    img = cv2.imread(img_path)
    img = cv2.resize(img, target_size)
    img = img / 255.0
    return np.expand_dims(img, axis=0)

# --- Get model input shape ---

def get_model_input_shape(model):
    # Returns input shape as (height, width)
    input_shape = model.input_shape
    if len(input_shape) == 4:  # (None, H, W, 3)
        return (input_shape[1], input_shape[2])
    elif len(input_shape) == 2:  # Flattened (None, features)
        return input_shape[1]
    else:
        raise ValueError("Unsupported model input shape: ", input_shape)

# --- Main Prediction Logic ---

# Path to the test image
test_image_path = r"D:/cancer detection/Te-no_0010.jpg"  # Replace with your image path

# Step 1: Classify cancer type
classifier_input_size = get_model_input_shape(classifier_model)
input_img = preprocess_image(test_image_path, classifier_input_size)
class_result = np.argmax(classifier_model.predict(input_img))
cancer_type = cancer_labels[class_result]
print(f"Cancer Type Detected: {cancer_type}")

# Step 2: Final prediction from specific CNN model
if cancer_type == 'Brain Tumor':
    model = brain_model
elif cancer_type == 'Skin Cancer':
    model = skin_model
else:
    model = breast_model

model_input = get_model_input_shape(model)

# If model expects flat input, reshape
if isinstance(model_input, int):
    flat_img = cv2.imread(test_image_path)
    flat_img = cv2.resize(flat_img, (150, 150))  # adjust this if needed
    flat_img = flat_img / 255.0
    flat_img = flat_img.reshape(1, -1)
    prediction = model.predict(flat_img)
else:
    input_img_pred = preprocess_image(test_image_path, model_input)
    prediction = model.predict(input_img_pred)

final_result = np.argmax(prediction)
print(f"Final Prediction: {prediction_labels[final_result]}")

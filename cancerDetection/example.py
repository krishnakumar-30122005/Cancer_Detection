
classifier_model = load_model(r"D:/cancer detection/cancerDetection/models/classifier_model.h5")
brain_model = load_model(r"D:/cancer detection/cancerDetection/models/train_cnn_brain.h5")
skin_model = load_model(r"D:/cancer detection/cancerDetection/models/train_cnn_skin.h5")
breast_model = load_model(r"D:/cancer detection/cancerDetection/models/train_cnn_breast.h5")

# Label mappings
cancer_labels = ['Brain Tumor', 'Skin Cancer', 'Breast Cancer']
prediction_labels = ['normal', 'cancer']

def preprocess_image(img_path, target_size):
    img = cv2.imread(img_path)
    img = cv2.resize(img, target_size)
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def get_model_input_shape(model):
    input_shape = model.input_shape
    if len(input_shape) == 4:
        return (input_shape[1], input_shape[2])
    elif len(input_shape) == 2:
        return input_shape[1]
    else:
        raise ValueError("Unsupported model input shape: ", input_shape)

# Test image path
test_image_path = r"D:/cancer detection/sample1.jpg"

# Step 1: Cancer type classification
classifier_input_size = get_model_input_shape(classifier_model)
input_img = preprocess_image(test_image_path, classifier_input_size)

classifier_pred = classifier_model.predict(input_img)
print("Classifier raw output:", classifier_pred)
class_result = np.argmax(classifier_pred)
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

if isinstance(model_input, int):  # Flattened input
    flat_img = cv2.imread(test_image_path)
    flat_img = cv2.resize(flat_img, (150, 150))  # Adjust if your model expects another size
    flat_img = flat_img / 255.0
    flat_img = flat_img.reshape(1, -1)
    prediction = model.predict(flat_img)
else:
    input_img_pred = preprocess_image(test_image_path, model_input)
    prediction = model.predict(input_img_pred)

final_result = np.argmax(prediction)
confidence = prediction[0][final_result]
print(f"Prediction probabilities: {prediction}")
print(f"Final Prediction: {prediction_labels[final_result]} with confidence: {confidence:.2f}")

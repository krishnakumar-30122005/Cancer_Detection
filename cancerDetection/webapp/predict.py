@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No file part in the request", 400

    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400

    image_path = os.path.join("static", file.filename)
    file.save(image_path)

    # Step 1: Detect Cancer Type using classifier model (CNN)
    img = Image.open(image_path).convert('RGB')
    img = img.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 150, 150, 3)

    predicted_type = np.argmax(classifier_model.predict(img_array))

    cancer_type = ""
    if predicted_type == 0:
        model = brain_model
        cancer_type = "Brain Tumor"
    elif predicted_type == 1:
        model = breast_model
        cancer_type = "Breast Cancer"
    else:
        model = skin_model
        cancer_type = "Skin Cancer"

    # Step 2: Predict Benign/Malignant using specific model
    prediction = model.predict(img_array)
    prediction_class = np.argmax(prediction)
    result = "Benign" if prediction_class == 0 else "Malignant"

    return render_template('result.html', prediction=f"{cancer_type} - {result}")

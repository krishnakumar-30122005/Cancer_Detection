from flask import Flask, render_template, request
import numpy as np
from keras.models import load_model
from PIL import Image
import io
from flask import send_file
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io

app = Flask(__name__)

# Load models
classifier_model = load_model("D:/cancer detection/cancerDetection/models/classifier_model.h5")
brain_model = load_model("D:/cancer detection/cancerDetection/models/train_cnn_brain.h5")
skin_model = load_model("D:/cancer detection/cancerDetection/models/train_cnn_skin.h5")
breast_model = load_model("D:/cancer detection/cancerDetection/models/train_cnn_breast.h5")

# Preprocess for classifier (expects 150x150x3 input)
def preprocess_for_classifier(img):
    img = img.resize((150, 150))
    img = np.array(img)
    if img.shape[-1] == 4:
        img = img[..., :3]
    img = img / 255.0
    return np.expand_dims(img, axis=0)  # Shape: (1, 150, 150, 3)

# Preprocess for CNN models (expects 224x224x3 input)
def preprocess_for_cnn(img):
    img = img.resize((224, 224))
    img = np.array(img)
    if img.shape[-1] == 4:
        img = img[..., :3]
    img = img / 255.0
    return np.expand_dims(img, axis=0)

@app.route('/')
def how_it_works():
    return render_template('how_it_works.html')

@app.route('/download_report')
def download_report():
    cancer_type = request.args.get('cancer_type')
    result = request.args.get('result')

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    pdf.setTitle("Cancer Prediction Report")

    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(100, 750, "Cancer Detection Report")

    pdf.setFont("Helvetica", 12)
    pdf.drawString(100, 700, f"Cancer Type: {cancer_type}")
    pdf.drawString(100, 680, f"Diagnosis: {result}")

    pdf.drawString(100, 640, "This report was generated automatically by the AI prediction system.")

    pdf.save()
    buffer.seek(0)

    return send_file(buffer, as_attachment=True, download_name="cancer_report.pdf", mimetype='application/pdf')

@app.route('/Upload')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part in the request'
    
    file = request.files['file']
    if file.filename == '':
        return '     No file selected'
    
    try:
        image = Image.open(io.BytesIO(file.read()))
        
        # Step 1: Predict cancer type
        clf_input = preprocess_for_classifier(image)
        cancer_type_pred = classifier_model.predict(clf_input)
        cancer_type = np.argmax(cancer_type_pred)

        cancer_label = {0: "Brain Tumor", 1: "Breast Cancer", 2: "Skin Cancer"}
        predicted_type = cancer_label.get(cancer_type, "Unknown")

        # Step 2: Predict benign/malignant
        cnn_input = preprocess_for_cnn(image)
        if cancer_type == 0:
            final_pred = brain_model.predict(cnn_input)
        elif cancer_type == 1:
            final_pred = breast_model.predict(cnn_input)
        elif cancer_type == 2:
            final_pred = skin_model.predict(cnn_input)
        else:
            return "Unrecognized cancer type"

        final_result = "Ubnoraml(Cancerous)" if final_pred[0][0] > 0.5 else "Normal(Non Cancerous)"

        return render_template("predict.html", cancer_type=predicted_type, result=final_result)

    except Exception as e:
        return f"Error during prediction: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)

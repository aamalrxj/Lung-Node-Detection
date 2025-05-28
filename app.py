from flask import Flask, request, render_template, send_from_directory 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os



app = Flask(__name__)

# Load the trained model from the models folder
model = load_model(os.path.join('models', 'lung_nodule_detection_model.h5'))

# Define image size for the model
img_height, img_width = 224, 224

# Class descriptions
class_descriptions = {
    0: "Nodule Detected:The CT scan contains Benign Nodule.\n This type of nodule is typically non-cancerous and may not pose a health risk.",
    1: "Nodule Detected:The CT scan contains Malignant Nodule.\n This type of nodule is indicative of potential cancer and requires further evaluation.",
    2: "No nodule detected.\n No suspicious nodules detected in the scan, indicating no immediate concern."
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # ===== NEW: Collect patient data =====
    patient_data = {
        'name': request.form.get('name'),
        'age': request.form.get('age'),
        'phone': request.form.get('phone'),
        'email': request.form.get('email')
    }

    if 'file' not in request.files:
        return render_template('index.html', error="No file part")
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="No selected file")
    if file:
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)

        # Load and preprocess the image
        img = image.load_img(file_path, target_size=(img_height, img_width))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]

        # Prepare prediction result
        predicted_label = class_descriptions[predicted_class]

        # ===== MODIFIED: Pass patient_data to template =====
        return render_template('result.html', 
                            prediction=predicted_label,
                            file_path=file.filename,
                            patient=patient_data)  # Added patient_data

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory('uploads', filename)

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)

from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load the model
model = load_model('potato_classification_model2.h5')

# Function to classify the image
def classify_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize as per your model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize if needed
    prediction = model.predict(img_array)
    return "Rotten" if prediction[0][0] < 0.5 else "Healthy"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            result = classify_image(file_path)
            return render_template('index.html', result=result, image_path=file_path)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

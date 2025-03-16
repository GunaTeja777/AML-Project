from flask import Flask, render_template, request, redirect, url_for, flash
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "deepfake_detection_secret_key"
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the deepfake detection model
model = None

def load_model():
    global model
    try:
        model = tf.keras.models.load_model(r"C:\Users\tejag\OneDrive\Desktop\AIML project\AML-Project\deepfake_EfficientNet.h5")
        model = tf.keras.models.load_model(r"C:\Users\tejag\OneDrive\Desktop\AIML project\AML-Project\deepfake_detector_final.h5")
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess the image for model prediction"""
    img = Image.open(image_path)
    img = img.resize(target_size)
    img = np.array(img) / 255.0  # Normalize to [0,1]
    return np.expand_dims(img, axis=0)  # Add batch dimension

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_deepfake():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        if model is None:
            flash('Model not loaded. Please try again later.')
            return redirect(url_for('index'))
        
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Preprocess image and make prediction
        try:
            processed_img = preprocess_image(file_path)
            prediction = model.predict(processed_img)
            
            # Interpret the prediction (this depends on your model output format)
            # Assuming a binary classification where higher = more likely to be fake
            score = float(prediction[0][0])
            probability = score * 100
            result = "REAL" if score > 0.5 else "FALSE"

            
            return render_template('result.html', 
                                  image_file=filename, 
                                  result=result,
                                  probability=probability)
        except Exception as e:
            flash(f'Error during processing: {e}')
            return redirect(url_for('index'))
    
    flash('Invalid file type. Please upload an image file (PNG, JPG, JPEG).')
    return redirect(url_for('index'))

if __name__ == '__main__':
    load_model()
    app.run(debug=True)

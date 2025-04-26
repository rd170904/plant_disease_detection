import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Load trained model
MODEL_PATH = os.path.abspath(r"C:\Users\renuk\Desktop\plantdisease\plant_disease_model.h5")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

# Define all possible class labels (Ensure it matches your model output)
class_labels = [
    'Tomato Healthy', 'Tomato Bacterial Spot', 'Tomato Early Blight', 
    'Tomato Late Blight', 'Tomato Leaf Mold', 'Tomato Septoria Leaf Spot', 
    'Tomato Spider Mites', 'Tomato Target Spot', 'Tomato Mosaic Virus', 
    'Tomato Yellow Leaf Curl Virus', 'Potato Healthy', 'Potato Early Blight',
    'Potato Late Blight', 'Raspberry Leaf', 'Strawberry Leaf Scorch', 
    'Corn Common Rust', 'Corn Gray Leaf Spot', 'Corn Healthy', 
    'Corn Northern Leaf Blight', 'Pepper Bell Healthy',
    'Pepper Bell Bacterial Spot', 'Soybean Healthy', 'Apple Scab', 
    'Apple Black Rot', 'Apple Cedar Rust', 'Grape Black Rot', 
    'Grape Esca (Black Measles)', 'Grape Leaf Blight'
]  # Update with actual 28 classes
  # Adjust based on model output

# List of healthy class labels
healthy_labels = {"Tomato Healthy", "Potato Healthy", "Corn Healthy", "Pepper Bell Healthy"}

# Ensure upload folder exists
UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    """Check if uploaded file has a valid image extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """Preprocess the image for model prediction."""
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((150, 150))  # Ensure model input size
        img = np.array(img) / 255.0  # Normalize pixel values
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        return img
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

@app.route("/", methods=["GET"])
def home():
    """Render the home page with an upload form."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Handle image uploads and make predictions."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Upload PNG, JPG, or JPEG"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    # Preprocess and predict
    img = preprocess_image(file_path)
    if img is None:
        return jsonify({"error": "Invalid image format"}), 400

    try:
        predictions = model.predict(img)

        # Apply softmax if not applied in model
        if "softmax" not in str(model.layers[-1].activation):
            probabilities = tf.nn.softmax(predictions[0]).numpy()
        else:
            probabilities = predictions[0]  # Already softmax

        # Get class with highest probability
        predicted_class = np.argmax(probabilities)
        confidence = round(float(np.max(probabilities)) * 100, 2)  # Convert to percentage

        # Ensure valid class label
        predicted_label = class_labels[predicted_class] if 0 <= predicted_class < len(class_labels) else "Unknown Disease"

        # Modify: Ignore healthy labels
        if predicted_label in healthy_labels:
            result = {
                "class": "No Disease Detected",
                "confidence": confidence
            }
        else:
            result = {
                "class": predicted_label,
                "confidence": confidence
            }

        return jsonify(result)

    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({"error": "Prediction failed"}), 500

if __name__ == "__main__":
    app.run(debug=True)

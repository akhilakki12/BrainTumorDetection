from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# -----------------------------
# Initialize Flask App
# -----------------------------
app = Flask(__name__)

# -----------------------------
# Load Model (IMPORTANT)
# -----------------------------
model=load_model("final_model.keras")  # Ensure this path is correct and the model file exists


# -----------------------------
# Class Labels (MUST match training order)
# -----------------------------
class_labels = ['glioma', 'meningioma','notumor','pituitary']
inverse_label_map = {i: label for i, label in enumerate(class_labels)}

# -----------------------------
# Upload Folder Setup
# -----------------------------
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# -----------------------------
# Prediction Function
# -----------------------------
def detect_and_predict(img_path, model, image_size=128):
    try:
        img = load_img(img_path, target_size=(image_size, image_size))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        predicted_class_index = int(np.argmax(predictions))
        confidence_score = float(np.max(predictions))

        predicted_label = inverse_label_map[predicted_class_index]

        if predicted_label == "notumor":
            result = "No Tumor"
        else:
            result = f"Tumor: {predicted_label}"

        return result, confidence_score

    except Exception as e:
        return f"Error: {str(e)}", 0.0


# -----------------------------
# Main Route
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]

        if file:
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)

            result, confidence = detect_and_predict(file_path, model)

            return render_template(
                "index.html",
                result=result,
                confidence=f"{confidence * 100:.2f}%",
                file_path=file.filename
            )

    return render_template("index.html", result=None)


# -----------------------------
# Serve Uploaded Images
# -----------------------------
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)

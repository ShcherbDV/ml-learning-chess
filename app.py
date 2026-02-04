import os


import tensorflow as tf
from flask import Flask, request, render_template

from classifier import classify
app = Flask(__name__)

STATIC_FOLDER = "static"
UPLOAD_FOLDER = os.path.join(STATIC_FOLDER, "uploads")

cnn_model = tf.keras.models.load_model(STATIC_FOLDER + "/models/" + "chess_model.h5")

@app.route("/")
def home():
    return "<p>Hello, please go to '/classify' to predict chess figure</p>"


@app.post("/classify")
def upload_file():
    if "file" not in request.files:
        return {"error": "No file in request"}, 400

    file = request.files["file"]

    if file.filename == "":
        return {"error": "No selected file"}, 400

    upload_image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(upload_image_path)

    label, confidence = classify(cnn_model, upload_image_path)

    return {
        "figure": label,
        "confidence": round(confidence * 100, 2),
    }


@app.get("/ui")
def ui():
    return render_template("index.html")


@app.post("/classify-ui")
def classify_ui():
    file = request.files.get("file")

    if not file or file.filename == "":
        return render_template("index.html", error="No file selected")

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(image_path)

    label, confidence = classify(cnn_model, image_path)

    return render_template(
        "index.html",
        figure=label,
        confidence=round(confidence * 100, 2)
    )


if __name__ == '__main__':
    app.debug = True
    app.run(debug=True)

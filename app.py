"""
app.py  -  Flask backend for the Brain Tumor Detection System.
This is the file you RUN. Start it with:  python app.py
Then open the link it prints (http://127.0.0.1:5000) in your browser.
"""

import os
import sqlite3
from datetime import datetime

import numpy as np
import cv2
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input

# ---------- Basic setup ----------
app = Flask(__name__)
UPLOAD_FOLDER = os.path.join("static", "uploads")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = os.path.join("model", "brain_tumor_model.h5")
DB_PATH = "database.db"
IMG_SIZE = 224  # the size the model expects (must match training)

# ---------- Load the trained model ONCE at startup ----------
# Loading is slow, so we do it here, not on every request.
model = load_model(MODEL_PATH)
print("Model loaded successfully.")


# ---------- Database setup ----------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS predictions (
               id        INTEGER PRIMARY KEY AUTOINCREMENT,
               filename  TEXT,
               result    TEXT,
               confidence REAL,
               created_at TEXT
           )"""
    )
    conn.commit()
    conn.close()


def save_prediction(filename, result, confidence):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO predictions (filename, result, confidence, created_at) VALUES (?, ?, ?, ?)",
        (filename, result, round(confidence, 2), datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    )
    conn.commit()
    conn.close()


# ---------- Image preprocessing ----------
# This MUST match how images were prepared during training.
# VGG16 requires its own preprocess_input (mean subtraction per channel).
def preprocess(image_path):
    img = cv2.imread(image_path)              # read image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV loads as BGR, convert to RGB
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # resize to 224x224
    img = np.expand_dims(img, axis=0).astype("float32")  # add batch dim -> (1, 224, 224, 3)
    img = preprocess_input(img)              # VGG16-specific normalization
    return img


# ---------- Routes (the pages of the app) ----------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # 1. Get the uploaded file
    if "file" not in request.files or request.files["file"].filename == "":
        return redirect(url_for("index"))

    file = request.files["file"]
    filename = file.filename
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    # 2. Preprocess and predict
    img = preprocess(save_path)
    prob = float(model.predict(img)[0][0])  # model outputs a single number 0-1

    # 3. Interpret the result
    # > 0.5 means tumor. Confidence is how sure the model is about its answer.
    if prob > 0.5:
        result = "Tumor Detected"
        confidence = prob * 100
    else:
        result = "No Tumor"
        confidence = (1 - prob) * 100

    # 4. Save to database
    save_prediction(filename, result, confidence)

    # 5. Show the result page
    return render_template(
        "result.html",
        result=result,
        confidence=round(confidence, 2),
        image_file=filename,
    )


@app.route("/history")
def history():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT filename, result, confidence, created_at FROM predictions ORDER BY id DESC"
    ).fetchall()
    conn.close()
    return render_template("history.html", rows=rows)


# ---------- Start the server ----------
if __name__ == "__main__":
    init_db()
    app.run(debug=True)

# Brain Tumor Detection System — Web App

A Flask web app that loads a trained deep-learning model and predicts whether an
uploaded MRI scan shows a brain tumor, with a confidence score and prediction history.

---

## What each file does

| File / Folder | What it is |
|---|---|
| `app.py` | The main backend. This is the file you RUN. |
| `create_model.py` | Builds a model file so the app can start (see step 4). |
| `model/brain_tumor_model.h5` | The trained model the app loads. |
| `templates/` | The HTML pages (upload, result, history). |
| `static/style.css` | Styling. |
| `static/script.js` | Image preview before upload. |
| `static/uploads/` | Uploaded MRI images get saved here. |
| `database.db` | SQLite database (created automatically on first run). |
| `requirements.txt` | The Python libraries you need. |

---

## Step-by-step setup (total beginner)

### 1. Install Python
Download Python 3.10 or 3.11 from python.org. During install, TICK
"Add Python to PATH". (TensorFlow does not yet support the very newest versions,
so avoid 3.12+.)

### 2. Install VS Code
Download from code.visualstudio.com. Open this project folder in it:
File → Open Folder → select `brain-tumor-detection`.

### 3. Open a terminal and install the libraries
In VS Code: Terminal → New Terminal. Then run:

```
python -m venv venv
```

Activate it:
- Windows:  `venv\Scripts\activate`
- Mac/Linux: `source venv/bin/activate`

Then install everything:

```
pip install -r requirements.txt
```

(This downloads TensorFlow and may take several minutes — that's normal.)

### 4. Make sure you have a model file
The app needs `model/brain_tumor_model.h5` to start.
- **Access the trained model.h5 from the below link**
  https://www.dropbox.com/scl/fi/k1ol5lhjbqvj8zvsf2aq5/brain_tumor_model.h5?rlkey=ray4bz5jq8es6s1kkdrkrgfgz&st=8z1ckz5v&dl=0
  
- **Place the file in**
  model/brain_tumor_model.h5
  
- **Quick test (random predictions):** run `python create_model.py`.
  This lets you see the whole app working end-to-end immediately, but the
  predictions are meaningless until you do the next part.

- **Real predictions:** see "Getting a trained model" below, then drop the
  downloaded `.h5` into the `model/` folder with the name `brain_tumor_model.h5`.

### 5. Run the app
```
python app.py
```
You'll see a line like:  `Running on http://127.0.0.1:5000`
Hold Ctrl and click that link (or paste it into your browser).

### 6. Use it
Upload an MRI image → get the result + confidence → check the History page.
To stop the app, press Ctrl + C in the terminal.

---

## Getting a trained model (for real predictions)

You chose to use a ready-made model, so you don't train anything. Options:

1. **Kaggle:** Search Kaggle for "brain tumor detection model h5" or
   "brain tumor VGG16". Many notebooks share a downloadable trained `.h5`.
   Download it, rename it to `brain_tumor_model.h5`, put it in `model/`.

2. **GitHub:** Search GitHub for "brain tumor detection flask h5". Several
   repos include a pre-trained model file you can download directly.

IMPORTANT: whatever model you download must expect **224 x 224 RGB** input and
output a **single sigmoid value** (tumor probability). If a model you find uses a
different input size or multiple output classes, tell me which one and I'll
adjust `app.py` to match it — that's the only part that needs changing.

---

## How a prediction flows through the system

1. You upload an MRI on the web page.
2. `app.py` saves the image and preprocesses it (resize to 224x224, normalize).
3. The model runs `predict()` and returns a number between 0 and 1.
4. Above 0.5 = Tumor, below = No Tumor. The number becomes a confidence %.
5. The result is saved to the SQLite database and shown on the result page.
6. The History page lists every past prediction.

---

## Common errors

- **"No module named tensorflow"** → your virtual environment isn't active, or
  you skipped `pip install -r requirements.txt`.
- **"Could not load model"** → you don't have `model/brain_tumor_model.h5` yet.
  Run `python create_model.py` or add a downloaded model.
- **Page won't open** → make sure the terminal still shows the app running and
  you used the exact `http://127.0.0.1:5000` address.

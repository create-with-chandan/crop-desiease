# CropAI production scanner

CropAI is a Flask and TensorFlow crop-disease classifier with an HTML frontend. The browser never invents a diagnosis: predictions come from the trained model served by `/api/predict`.

The classifier is dataset-driven, so it supports vegetables, field crops, fruit crops, and ornamental plants when their labeled classes are included in training. It does not claim to identify an unseen disease class: unknown or low-confidence images are rejected.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000/`.

The scanner runs the trained model directly in the browser with TensorFlow.js,
so the hosted site (for example on Netlify) works without a Python server.
The Flask server is optional and only needed for the `/api/predict` endpoint.
Do not open `scanner.html` by double-clicking it; serve the folder over HTTP
(for example with `python app.py` or `netlify dev`) so the model files can load.

The trained model and its matching labels are included in Git so a deployment
can start ready to serve predictions. Keep the training dataset out of Git.
When replacing the model, update both `models/crop_disease.keras` and
`models/labels.json` together.

To train with PlantVillage, first obtain the dataset from its source and save the
archive as `datasets/plantvillage.zip`. Export its labeled image folders with:

```powershell
python prepare_plantvillage.py --archive datasets\plantvillage.zip --output training-data
```

Then train the model:

```powershell
python train.py --data training-data --output models --epochs 12
```

Then export the browser model used by the scanner (needs only `h5py` and `numpy`):

```powershell
python export_tfjs.py --model models\crop_disease.keras --output models\tfjs
```

After training, restart the server and check `http://127.0.0.1:5000/api/health`
for `"model_loaded": true`.

## Train the classifier

Prepare a directory with one subdirectory per class. Add as many crop and plant classes as your labeled dataset supports. Class names should identify both plant and condition, for example:

```text
training-data/
  Tomato___Early_blight/
  Tomato___healthy/
  Potato___Late_blight/
  Potato___healthy/
  Cucumber___Powdery_mildew/
  Pepper___Bacterial_spot/
  Apple___healthy/
```

Then run:

```powershell
python train.py --data training-data --output models --epochs 12
```

This creates `models/crop_disease.keras` and `models/labels.json`. Each class should have at least five images; hundreds or thousands per class are strongly recommended for production. Restart `app.py` after training. The API uses OpenCV to reject obvious non-plant images, then runs TensorFlow inference and rejects low-confidence results.

## Production notes

- **Deploy to Render:** push this project to GitHub with both files in `models/`,
  then create a Render Blueprint from the repository. `render.yaml` configures
  the Python runtime, production Gunicorn server, and health check. Choose an
  instance with enough memory for TensorFlow; the included Standard plan is a
  paid Render instance. The deployment URL serves the site and scanner on one
  origin.
- Set `CROP_MIN_CONFIDENCE` in the host's environment to tune the prediction
  rejection threshold; the default is `0.55`.
- Use a representative, licensed dataset covering every crop and disease class you intend to support.
- “All plants” is not a finite promise for one classifier. Add new labeled classes and retrain whenever the supported crop or disease scope changes.
- Measure precision, recall, confusion matrices, and out-of-distribution rejection on a held-out test set before deployment.
- Keep `CROP_MIN_CONFIDENCE` conservative and have an agronomist review treatment recommendations.
- Gunicorn runs one worker to avoid loading a duplicate TensorFlow model into
  memory. The application limits uploads to 8 MB.
- If either trained model artifact is missing or cannot load, the service
  reports `model_not_ready` and returns `503` rather than making up a result.
- GitHub Pages can host the static HTML/CSS/JavaScript pages, but it cannot run the Flask API required by the scanner. Deploy the Flask application separately to use image predictions.

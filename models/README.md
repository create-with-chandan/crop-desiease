Place the trained TensorFlow artifacts in this directory:

- `crop_disease.keras`
- `labels.json`

Train them from a folder dataset with:

```powershell
python train.py --data path\to\dataset --output models
```

Each child folder is one class. Use names such as `Tomato___Early_blight` and `Tomato___healthy` so the API can display the plant and disease correctly.

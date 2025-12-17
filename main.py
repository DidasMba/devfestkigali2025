from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = FastAPI(title="Agriculture AI Demo")

# Charger modèle
model = load_model('agri_model.h5')

@app.post("/predict")
async def predict_leaf(file: UploadFile = File(...)):
    img = image.load_img(file.file, target_size=(128,128))
    x = image.img_to_array(img)/255.0
    x = np.expand_dims(x, axis=0)
    pred = model.predict(x)
    label = "Diseased" if pred[0][0] > 0.5 else "Healthy"
    return {"prediction": label}

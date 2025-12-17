# Agri Model – Crop Analysis

This model is trained to help with agriculture insights.

## Details

- **Format:** .h5
- **Owner:** Didas Mbarushimana
- **Use case:** Predicting crop health from images

## Usage

You can download and use this model with TensorFlow / Keras:

```python
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model

model_path = hf_hub_download(
    repo_id="Didasmb/agri_model",
    filename="agri_model.h5"
)
model = load_model(model_path)
```

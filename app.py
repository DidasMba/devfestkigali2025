# import streamlit as st
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import numpy as np
# from PIL import Image

# # Titre de l'app
# st.title("🌾 Plant Disease Classifier")
# st.write("Upload an image of a leaf and I will classify it as Healthy or Diseased")

# # Upload de l'image
# uploaded_file = st.file_uploader("📸 Upload image", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Affiche l'image uploadée
#     img = Image.open(uploaded_file)
#     st.image(img, caption='Uploaded Image', use_column_width=True)

#     # Chargement du modèle CNN
#     model = load_model('agri_model.h5')

#     # Récupération automatique de la taille d'entrée attendue par le modèle
#     input_shape = model.input_shape  # par ex. (None, 150, 150, 3)
#     target_size = input_shape[1:3]   # (hauteur, largeur)

#     # Préparation de l'image pour le modèle
#     img = img.resize(target_size)
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)  # ajoute la dimension batch
#     img_array /= 255.0  # normalisation

#     # Prédiction
#     prediction = model.predict(img_array)[0][0]

#     # Affichage du résultat
#     if prediction > 0.5:
#         st.success("❌ Diseased Leaf")
#     else:
#         st.success("✅ Healthy Leaf")





import gradio as gr
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Charger le modèle une seule fois au démarrage
model = load_model("agri_model.h5")
input_shape = model.input_shape[1:3]  # taille attendue par le modèle (hauteur, largeur)

# Fonction de prédiction
def classify_leaf(img):
    if img is None:
        return "No image uploaded"

    # Préparer l'image
    img = img.resize(input_shape)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Prédiction
    prediction = model.predict(img_array)[0][0]

    # Résultat
    if prediction > 0.5:
        return "❌ Diseased Leaf"
    else:
        return "✅ Healthy Leaf"

# Interface Gradio
demo = gr.Interface(
    fn=classify_leaf,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="🌾 Plant Disease Classifier",
    description="Upload an image of a leaf and I will classify it as Healthy or Diseased"
)

demo.launch()


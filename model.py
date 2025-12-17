from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Préparation des images
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    'sample_images',          # ton dossier contenant les images classées par sous-dossier
    target_size=(150, 150),
    batch_size=4,
    class_mode='binary',      # Healthy / Diseased
    subset='training'
)

val_gen = datagen.flow_from_directory(
    'sample_images',
    target_size=(150, 150),
    batch_size=4,
    class_mode='binary',
    subset='validation'
)

# Création du modèle CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),   # on a besoin de flatten juste avant les densités
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # sortie binaire
])

# Compilation
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entraînement
model.fit(train_gen, validation_data=val_gen, epochs=3)

# Sauvegarde
model.save('agri_model.h5')
print("🎉 Modèle CNN entraîné et sauvegardé sous 'agri_model.h5'")


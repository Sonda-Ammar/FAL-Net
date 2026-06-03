# data_loader.py
import os
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import os
import numpy as np
from PIL import Image


def load_images_from_directory(directory, label):
    images = []
    labels = []

    # Parcours de tous les fichiers dans le répertoire
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg', '.bmp')):  # Vérifier les extensions d'image
                image_path = os.path.join(root, file)
                try:
                    image = Image.open(image_path).convert('RGB')  # Convertir en RGB
                    image = image.resize((224, 224))  # Redimensionner l'image
                    image = np.array(image)  # Convertir en tableau NumPy
                    images.append(image)
                    labels.append(label)
                except Exception as e:  # Gestion des erreurs (par exemple, image corrompue)
                    print(f"Erreur lors de l'ouverture de {image_path}: {e}")
                    continue  # Continuer avec l'image suivante en cas d'erreur

    return np.array(images), np.array(labels)


def prepare_data(real_dir_train, fake_dir_train, real_dir_test, fake_dir_test):
    # Charger les images réelles (label 0 pour "real")
    real_images, real_labels = load_images_from_directory(real_dir_train, label=0)

    # Charger les images fausses (label 1 pour "fake")
    fake_images, fake_labels = load_images_from_directory(fake_dir_train, label=1)

    # Combiner les données
    all_images_train = np.concatenate([real_images, fake_images], axis=0)
    all_labels_train = np.concatenate([real_labels, fake_labels], axis=0)
    all_labels_train = to_categorical(all_labels_train, num_classes=2)

    # Charger les images réelles de test (label 0 pour "real")
    real_images_test, real_labels_test = load_images_from_directory(real_dir_test, label=0)

    # Charger les images fausses de test (label 1 pour "fake")
    fake_images_test, fake_labels_test = load_images_from_directory(fake_dir_test, label=1)

    # Combiner les données de test
    all_images_test = np.concatenate([real_images_test, fake_images_test], axis=0)
    all_labels_test = np.concatenate([real_labels_test, fake_labels_test], axis=0)
    all_labels_test = to_categorical(all_labels_test, num_classes=2)
    # Créer les générateurs de données
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=10,
        width_shift_range=5,
        height_shift_range=5,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    train_generator = train_datagen.flow(
        all_images_train,  # Images chargées
        all_labels_train,  # Étiquettes
        batch_size=16,
        shuffle=True
    )
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow(
        all_images_test,  # Images chargées
        all_labels_test,  # Étiquettes
        batch_size=16,
        shuffle=False
    )
    return train_generator, test_generator, all_labels_test
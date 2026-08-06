# data_loader.py
import os
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical


def load_images_from_directory(directory, label):
    images = []
    labels = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_path = os.path.join(root, file)
                try:
                    image = Image.open(image_path).convert('RGB')
                    image = image.resize((224, 224))
                    image = np.array(image)
                    images.append(image)
                    labels.append(label)
                except Exception as e:
                    print(f"Erreur lors de l'ouverture de {image_path}: {e}")
                    continue

    return np.array(images), np.array(labels)


def prepare_train_val_data(real_dir_train, fake_dir_train, val_split=0.15, seed=42, batch_size=16):
    """
    Charge le train set complet, puis effectue un split interne stratifie train/val.
    Le test set n'est jamais touche par cette fonction (utiliser prepare_test_data).

    Encodage : 0 = Live (real), 1 = Fake (spoof)
    """
    real_images, real_labels = load_images_from_directory(real_dir_train, label=0)
    fake_images, fake_labels = load_images_from_directory(fake_dir_train, label=1)

    all_images_train = np.concatenate([real_images, fake_images], axis=0)
    all_labels_train = np.concatenate([real_labels, fake_labels], axis=0)

    # --- Split stratifie par classe (evite un desequilibre live/fake entre train et val) ---
    rng = np.random.RandomState(seed)

    real_n = len(real_labels)
    fake_n = len(fake_labels)

    real_idx = rng.permutation(real_n)
    fake_idx = rng.permutation(fake_n) + real_n  # offset car fake concatene apres real

    real_val_size = int(real_n * val_split)
    fake_val_size = int(fake_n * val_split)

    val_idx = np.concatenate([real_idx[:real_val_size], fake_idx[:fake_val_size]])
    train_idx = np.concatenate([real_idx[real_val_size:], fake_idx[fake_val_size:]])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    train_images, train_labels = all_images_train[train_idx], all_labels_train[train_idx]
    val_images, val_labels = all_images_train[val_idx], all_labels_train[val_idx]

    train_labels_cat = to_categorical(train_labels, num_classes=2)
    val_labels_cat = to_categorical(val_labels, num_classes=2)

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=10,
        width_shift_range=5,
        height_shift_range=5,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow(
        train_images, train_labels_cat, batch_size=batch_size, shuffle=True, seed=seed
    )
    val_generator = val_datagen.flow(
        val_images, val_labels_cat, batch_size=batch_size, shuffle=False
    )

    return train_generator, val_generator


def prepare_test_data(real_dir_test, fake_dir_test, batch_size=16):
    """
    Charge le test set uniquement. Jamais vu pendant l'entrainement (ni train ni val).

    Encodage : 0 = Live (real), 1 = Fake (spoof)
    """
    real_images_test, real_labels_test = load_images_from_directory(real_dir_test, label=0)
    fake_images_test, fake_labels_test = load_images_from_directory(fake_dir_test, label=1)

    all_images_test = np.concatenate([real_images_test, fake_images_test], axis=0)
    all_labels_test = np.concatenate([real_labels_test, fake_labels_test], axis=0)
    all_labels_test_cat = to_categorical(all_labels_test, num_classes=2)

    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow(
        all_images_test, all_labels_test_cat, batch_size=batch_size, shuffle=False
    )

    return test_generator, all_labels_test_cat

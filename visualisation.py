import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Charger le modèle
model = load_model('best_model_with_attention_GreenBit_Digital_Persona.h5')
print("Modèle chargé avec succès.")
print(model.summary())

# Extraire la dernière couche convolutive
last_conv_layer = model.get_layer('conv5_block3_out')  # Adaptez le nom de la couche

# Créer un modèle pour les activations et les prédictions
grad_model = Model(inputs=model.inputs, outputs=[last_conv_layer.output, model.output])

# Charger et prétraiter l'image
image_path = '../Fingerprint/2015/Fingerprint/LivDet2015/Training/GreenBit/Live/002_0_7.png'
image = cv2.imread(image_path)
if image is None:
    raise ValueError("L'image n'a pas pu être chargée. Vérifiez le chemin.")
input_shape = model.input_shape[1:3]  # Taille d'entrée attendue par le modèle
image = cv2.resize(image, input_shape)
image = image.astype('float32') / 255.0  # Normaliser entre 0 et 1
image = np.expand_dims(image, axis=0)   # Ajouter une dimension pour le batch

# Vérifier les prédictions
predictions = model.predict(image)
print("Prédictions du modèle :", predictions)

# Calculer la Grad-CAM
def compute_gradcam(image, model, last_conv_layer):
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        loss = predictions[:, 0]  # Supposons que la classe 0 est "fake"
    grads = tape.gradient(loss, conv_outputs)[0]
    output = conv_outputs[0]
    return output, grads

output, grads = compute_gradcam(image, model, last_conv_layer)
print("Shape des activations de la dernière couche convolutive :", output.shape)
print("Shape des gradients :", grads.shape)

# Générer la Grad-CAM
def generate_gradcam(output, grads):
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = np.zeros(output.shape[0:2], dtype=np.float32)
    for index, w in enumerate(weights):
        cam += w * output[:, :, index]
    cam = np.maximum(cam, 0)  # Appliquer ReLU
    cam = cam / np.max(cam)   # Normaliser entre 0 et 1
    return cam

cam = generate_gradcam(output, grads)
print("Shape de la Grad-CAM :", cam.shape)

# Afficher la Grad-CAM
def visualize_gradcam(image, cam):
    cam = cv2.resize(cam, (image.shape[1], image.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + image * 0.6
    plt.imshow(superimposed_img / 255.0)
    plt.show()

# Charger l'image originale pour l'affichage
original_image = cv2.imread(image_path)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # Convertir en RGB

# Afficher la Grad-CAM
visualize_gradcam(original_image, cam)
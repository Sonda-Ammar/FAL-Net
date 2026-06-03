# model.py
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
import tensorflow as tf

class LoRALayer(layers.Layer):
    def __init__(self, original_layer, rank=8, **kwargs):
        super(LoRALayer, self).__init__(**kwargs)
        self.original_layer = original_layer
        self.rank = rank
        self.A = None
        self.B = None

    def build(self, input_shape):
        if isinstance(self.original_layer, layers.Conv2D):
            kernel_shape = self.original_layer.kernel_size + (self.original_layer.input_shape[-1], self.rank)
            self.A = self.add_weight(shape=kernel_shape, initializer="random_normal", trainable=True, name="lora_A")
            self.B = self.add_weight(shape=(self.original_layer.kernel_size[0], self.original_layer.kernel_size[1], self.rank, self.original_layer.filters),
                                     initializer="random_normal", trainable=True, name="lora_B")

        elif isinstance(self.original_layer, layers.Dense):
            input_dim = input_shape[-1]
            output_dim = self.original_layer.units
            self.A = self.add_weight(shape=(input_dim, self.rank), initializer="random_normal", trainable=True, name="lora_A")
            self.B = self.add_weight(shape=(self.rank, output_dim), initializer="random_normal", trainable=True, name="lora_B")

    def call(self, inputs, training=False):
        if isinstance(self.original_layer, layers.Conv2D):
            lora_update = tf.nn.conv2d(inputs, self.A, strides=1, padding='SAME')
            lora_update = tf.nn.conv2d(lora_update, self.B, strides=1, padding='SAME')

        elif isinstance(self.original_layer, layers.Dense):
            lora_update = tf.matmul(inputs, self.A)
            lora_update = tf.matmul(lora_update, self.B)

        return self.original_layer(inputs) + lora_update

    def get_config(self):
        config = super().get_config()
        config.update({
            "original_layer": layers.serialize(self.original_layer),  # Sérialisation de la couche originale
            "rank": self.rank
        })
        return config

    @classmethod
    def from_config(cls, config):
        config["original_layer"] = layers.deserialize(config["original_layer"])  # Désérialisation de la couche originale
        return cls(**config)



def attention_block(input_tensor, skip_tensor, filters):
    """
    Bloc d'attention pour U-Net.
    """
    g = layers.Conv2D(filters, (1, 1), padding='same')(input_tensor)
    x = layers.Conv2D(filters, (1, 1), padding='same')(skip_tensor)
    psi = layers.Add()([g, x])
    psi = layers.Activation('relu')(psi)
    psi = layers.Conv2D(1, (1, 1), padding='same', activation='sigmoid')(psi)
    attention_applied = layers.multiply([skip_tensor, psi])
    return attention_applied

'''def build_unet_with_attention(input_shape, num_classes):
    """
    Construit un U-Net avec mécanisme d'attention et LoRA.
    """
    inputs = layers.Input(input_shape)

    # Encoder (ResNet50 comme backbone)
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=inputs)

    # Appliquer LoRA sur certaines couches convolutives
    modified_layers = []
    for layer in base_model.layers:
        if isinstance(layer, layers.Conv2D):
            modified_layers.append(LoRALayer(layer, rank=8))
        else:
            modified_layers.append(layer)

    # Extraire les features de ResNet50
    s1 = base_model.get_layer('conv1_relu').output
    s2 = base_model.get_layer('conv2_block3_out').output
    s3 = base_model.get_layer('conv3_block4_out').output
    s4 = base_model.get_layer('conv4_block6_out').output
    bridge = base_model.get_layer('conv5_block3_out').output

    # Decoder avec attention
    d1 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(bridge)
    d1 = attention_block(d1, s4, 512)
    d1 = layers.concatenate([d1, s4])
    d1 = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(d1)

    d2 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(d1)
    d2 = attention_block(d2, s3, 256)
    d2 = layers.concatenate([d2, s3])
    d2 = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(d2)

    d3 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(d2)
    d3 = attention_block(d3, s2, 128)
    d3 = layers.concatenate([d3, s2])
    d3 = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(d3)

    d4 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(d3)
    d4 = attention_block(d4, s1, 64)
    d4 = layers.concatenate([d4, s1])
    d4 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(d4)

    # Couche finale avant la classification
    x = layers.GlobalAveragePooling2D()(d4)

    # Correction : Appliquer LoRA sur la couche Dense
    dense_layer = layers.Dense(512, activation="relu")
    x = LoRALayer(dense_layer, rank=8)(x)
    x = layers.Dropout(0.5)(x)

    # Couche finale de classification
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    # Créer le modèle
    model = models.Model(inputs, outputs)

    return model
'''
def build_unet(input_shape, num_classes, lora_rank=8, use_ssa=True, use_multiscale=True, freeze_backbone=False):
    """
    Construit un U-Net avec mécanisme d'attention et LoRA (flexible pour ablation).
    """
    inputs = layers.Input(input_shape)

    # Encoder (ResNet50 comme backbone)
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=inputs)
    
    if freeze_backbone:
        base_model.trainable = False
        print(">>> Backbone (ResNet50) has been FROZEN.")
    else:
        print(">>> Backbone (ResNet50) is TRAINABLE.")

    # Appliquer LoRA sur certaines couches convolutives (si lora_rank > 0)
    for layer in base_model.layers:
        if lora_rank > 0 and isinstance(layer, layers.Conv2D):
            # Note: Wrapping layers in-place in a functional model is tricky.
            # Usually we need to rebuild the graph. Here we follow the user's logic
            # but allow disabling it.
            pass 

    # Extraire les features de ResNet50
    s1 = base_model.get_layer('conv1_relu').output
    s2 = base_model.get_layer('conv2_block3_out').output
    s3 = base_model.get_layer('conv3_block4_out').output
    s4 = base_model.get_layer('conv4_block6_out').output
    bridge = base_model.get_layer('conv5_block3_out').output

    # Decoder
    # Stage 1
    d1_up = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(bridge)
    if use_ssa:
        d1_side = attention_block(d1_up, s4, 512)
    else:
        d1_side = s4
    
    if use_multiscale:
        d1 = layers.concatenate([d1_up, d1_side])
    else:
        d1 = d1_up # Single-scale path
    d1 = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(d1)

    # Stage 2
    d2_up = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(d1)
    if use_ssa:
        d2_side = attention_block(d2_up, s3, 256)
    else:
        d2_side = s3
        
    if use_multiscale:
        d2 = layers.concatenate([d2_up, d2_side])
    else:
        d2 = d2_up
    d2 = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(d2)

    # Stage 3
    d3_up = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(d2)
    if use_ssa:
        d3_side = attention_block(d3_up, s2, 128)
    else:
        d3_side = s2
        
    if use_multiscale:
        d3 = layers.concatenate([d3_up, d3_side])
    else:
        d3 = d3_up
    d3 = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(d3)

    # Stage 4
    d4_up = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(d3)
    if use_ssa:
        d4_side = attention_block(d4_up, s1, 64)
    else:
        d4_side = s1
        
    if use_multiscale:
        d4 = layers.concatenate([d4_up, d4_side])
    else:
        d4 = d4_up
    d4 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(d4)

    # Couche finale avant la classification
    x = layers.GlobalAveragePooling2D()(d4)

    # Appliquer LoRA sur la couche Dense (si lora_rank > 0)
    dense_layer = layers.Dense(512, activation="relu")
    if lora_rank > 0:
        x = LoRALayer(dense_layer, rank=lora_rank)(x)
    else:
        x = dense_layer(x)
    x = layers.Dropout(0.5)(x)

    # Couche finale de classification
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    # Créer le modèle
    model = models.Model(inputs, outputs)

    # If lora_rank > 0, we can also apply it to the encoder layers if needed
    # But the user's logic in lines 136-141 was incomplete (it didn't use modified_layers).
    # We maintain the classification-head-only LoRA which is what's discussed in the paper.

    return model

"""TensorFlow/Keras model builders for binary aerial classification."""
from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess_input


def build_augmentation() -> keras.Sequential:
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomFlip("vertical"),
            layers.RandomRotation(0.12),
            layers.RandomZoom(0.15),
            layers.RandomBrightness(0.15),
            layers.RandomContrast(0.1),
        ],
        name="data_augmentation",
    )


def build_custom_cnn(
    img_size: tuple[int, int] = (224, 224),
    dropout_rate: float = 0.35,
) -> keras.Model:
    """Small CNN with batch norm and dropout for Bird vs Drone."""
    inputs = keras.Input(shape=(*img_size, 3))
    aug = build_augmentation()
    x = aug(inputs)
    x = layers.Rescaling(1.0 / 255.0)(x)

    for filters in (32, 64, 128, 256):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.2)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="custom_cnn_bird_drone")
    return model


def build_efficientnet_transfer(
    img_size: tuple[int, int] = (224, 224),
    trainable_base_layers: int = 40,
    dropout_rate: float = 0.3,
) -> keras.Model:
    """EfficientNetB0 backbone with fine-tuning on the top layers."""
    inputs = keras.Input(shape=(*img_size, 3))
    aug = build_augmentation()
    x = aug(inputs)
    x = layers.Lambda(lambda img: efficientnet_preprocess_input(img))(x)

    base = EfficientNetB0(include_top=False, weights="imagenet")
    base.trainable = True
    for layer in base.layers[:-trainable_base_layers]:
        layer.trainable = False

    x = base(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="efficientnetb0_bird_drone")
    return model


def compile_binary_model(model: keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )

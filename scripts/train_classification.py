"""Train Custom CNN or EfficientNetB0 transfer model for Bird vs Drone."""
from __future__ import annotations

import os

# Quieter TensorFlow / oneDNN logs on Windows (must run before importing tensorflow).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf
from tensorflow import keras

from src import config
from src.models import build_custom_cnn, build_efficientnet_transfer, compile_binary_model
from src.yolo_to_classification import ensure_classification_from_yolo


def _to_binary(image, label):
    return image, tf.expand_dims(tf.cast(label, tf.float32), -1)


def make_datasets():
    ensure_classification_from_yolo()
    train_dir = config.CLASSIFICATION_ROOT / "TRAIN"
    val_dir = config.CLASSIFICATION_ROOT / "VALID"

    train_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        label_mode="int",
        seed=config.SEED,
    )
    val_ds = keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        label_mode="int",
        seed=config.SEED,
    )
    class_names = list(train_ds.class_names)
    train_ds = train_ds.map(_to_binary, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(_to_binary, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds, class_names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("cnn", "efficientnet"), required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    config.ensure_output_dirs()
    keras.utils.set_random_seed(config.SEED)

    train_ds, val_ds, class_names = make_datasets()
    meta = {"class_names": class_names, "img_size": list(config.IMG_SIZE), "model": args.model}
    (config.MODELS_DIR / "classification_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if args.model == "cnn":
        model = build_custom_cnn(img_size=config.IMG_SIZE)
        compile_binary_model(model, learning_rate=args.lr)
        ckpt_path = config.MODELS_DIR / "best_cnn.keras"
    else:
        model = build_efficientnet_transfer(img_size=config.IMG_SIZE)
        compile_binary_model(model, learning_rate=args.lr * 0.3)
        ckpt_path = config.MODELS_DIR / "best_efficientnet.keras"

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(filepath=str(ckpt_path), monitor="val_accuracy", save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
    ]

    t0 = time.perf_counter()
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)
    elapsed = time.perf_counter() - t0

    hist_path = config.MODELS_DIR / f"history_{args.model}.json"
    hist_path.write_text(json.dumps({k: [float(x) for x in v] for k, v in history.history.items()}, indent=2), encoding="utf-8")
    timing = {"model": args.model, "train_seconds": round(elapsed, 2), "epochs_ran": len(history.history["loss"])}
    (config.MODELS_DIR / f"timing_{args.model}.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")

    print(f"Saved checkpoint: {ckpt_path}")
    print(f"Training time: {elapsed:.1f}s ({timing['epochs_ran']} epochs)")


if __name__ == "__main__":
    main()

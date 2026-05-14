"""Evaluate a saved classifier on TEST: metrics, confusion matrix, curves."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tensorflow import keras

from src import config
from src.yolo_to_classification import ensure_classification_from_yolo


def load_test_ds():
    test_dir = config.CLASSIFICATION_ROOT / "TEST"
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Missing {test_dir}")
    ds = keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        label_mode="int",
        shuffle=False,
    )
    class_names = list(ds.class_names)

    def to_bin(image, label):
        return image, tf.expand_dims(tf.cast(label, tf.float32), -1)

    ds = ds.map(to_bin, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    return ds, class_names


def collect_predictions(model, ds):
    y_true, y_prob = [], []
    for images, labels in ds:
        preds = model.predict_on_batch(images)
        y_prob.extend(np.ravel(preds).tolist())
        y_true.extend(np.ravel(labels.numpy()).tolist())
    return np.array(y_true), np.array(y_prob)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=Path,
        default=config.MODELS_DIR / "best_efficientnet.keras",
        help="Path to .keras weights",
    )
    parser.add_argument("--compare-histories", action="store_true", help="Plot CNN vs EfficientNet val curves if JSONs exist")
    args = parser.parse_args()

    config.ensure_output_dirs()

    if args.compare_histories:
        cnn_hist = config.MODELS_DIR / "history_cnn.json"
        eff_hist = config.MODELS_DIR / "history_efficientnet.json"
        if not cnn_hist.is_file() or not eff_hist.is_file():
            print("Need history_cnn.json and history_efficientnet.json in outputs/models")
            return
        h_c = json.loads(cnn_hist.read_text(encoding="utf-8"))
        h_e = json.loads(eff_hist.read_text(encoding="utf-8"))
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(h_c.get("val_accuracy", []), label="CNN")
        ax[0].plot(h_e.get("val_accuracy", []), label="EfficientNetB0")
        ax[0].set_title("Validation accuracy")
        ax[0].set_xlabel("Epoch")
        ax[0].legend()
        ax[1].plot(h_c.get("val_loss", []), label="CNN")
        ax[1].plot(h_e.get("val_loss", []), label="EfficientNetB0")
        ax[1].set_title("Validation loss")
        ax[1].set_xlabel("Epoch")
        ax[1].legend()
        plt.tight_layout()
        out = config.FIGURES_DIR / "compare_val_curves.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"Saved {out}")
        return

    if not args.weights.is_file():
        raise FileNotFoundError(f"Model not found: {args.weights}. Train first with scripts/train_classification.py")

    ensure_classification_from_yolo()
    model = keras.models.load_model(args.weights)
    ds, class_names = load_test_ds()
    y_true, y_prob = collect_predictions(model, ds)
    y_pred = (y_prob >= 0.5).astype(int)

    print("Classification report (positive label = drone = 1):")
    print(classification_report(y_true, y_pred, target_names=class_names))

    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"F1-score (drone positive): {f1:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", color="w" if cm.max() > 0 and v > cm.max() / 2 else "k")
    fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    cm_path = config.FIGURES_DIR / f"confusion_matrix_{args.weights.stem}.png"
    fig.savefig(cm_path, dpi=120)
    plt.close(fig)
    print(f"Saved {cm_path}")

    metrics = {
        "weights": str(args.weights.resolve()),
        "accuracy": float((y_true == y_pred).mean()),
        "f1_drone_positive": float(f1),
        "classification_report": classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
        ),
    }
    metrics_path = config.MODELS_DIR / f"test_metrics_{args.weights.stem}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved {metrics_path}")

    stem_l = args.weights.stem.lower()
    if "cnn" in stem_l:
        hist_path = config.MODELS_DIR / "history_cnn.json"
    elif "efficientnet" in stem_l:
        hist_path = config.MODELS_DIR / "history_efficientnet.json"
    else:
        hist_path = None
    if hist_path and hist_path.is_file():
        h = json.loads(hist_path.read_text(encoding="utf-8"))
        fig2, ax2 = plt.subplots(1, 2, figsize=(9, 3.5))
        ax2[0].plot(h.get("accuracy", []), label="train")
        ax2[0].plot(h.get("val_accuracy", []), label="val")
        ax2[0].set_title("Accuracy")
        ax2[0].legend()
        ax2[1].plot(h.get("loss", []), label="train")
        ax2[1].plot(h.get("val_loss", []), label="val")
        ax2[1].set_title("Loss")
        ax2[1].legend()
        plt.tight_layout()
        curve_path = config.FIGURES_DIR / f"curves_{args.weights.stem}.png"
        fig2.savefig(curve_path, dpi=120)
        plt.close(fig2)
        print(f"Saved {curve_path}")


if __name__ == "__main__":
    main()

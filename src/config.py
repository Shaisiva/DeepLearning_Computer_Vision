"""Paths and constants for the aerial CV project."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# YOLOv8 dataset (images + labels). Default: repo folder below — on your machine this is:
# C:\Users\sivaelango.periyasam\source\repos\Guvi\05_DeepLearning_Computer_Vision\object_detection_Dataset
# Override with env AERIAL_OBJECT_DETECTION_ROOT if the dataset lives elsewhere.
_OBJECT_DETECTION_FALLBACK = ROOT / "object_detection_Dataset"
OBJECT_DETECTION_ROOT = Path(
    os.environ.get("AERIAL_OBJECT_DETECTION_ROOT", str(_OBJECT_DETECTION_FALLBACK))
).expanduser().resolve()
YOLO_DATA_YAML = OBJECT_DETECTION_ROOT / "data.yaml"

# Place your Bird/Drone classification tree here (folder layout differs from YOLO):
#   data/classification_dataset/TRAIN/{bird,drone}/...
#   data/classification_dataset/VALID/{bird,drone}/...
#   data/classification_dataset/TEST/{bird,drone}/...
CLASSIFICATION_ROOT = ROOT / "data" / "classification_dataset"

MODELS_DIR = ROOT / "outputs" / "models"
# Pretrained .pt files (e.g. yolov8n.pt); used by scripts/train_yolo.py when SSL blocks Ultralytics download.
YOLO_WEIGHTS_DIR = ROOT / "weights"
YOLO_OUTPUT_DIR = ROOT / "outputs" / "yolo"
FIGURES_DIR = ROOT / "outputs" / "figures"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

CLASS_NAMES = ("bird", "drone")


def ensure_output_dirs() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    YOLO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    YOLO_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

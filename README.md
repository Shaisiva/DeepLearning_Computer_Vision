# Aerial Bird vs Drone — classification & detection

Deep learning project: binary **image classification** (bird vs drone) with a **custom CNN** and **EfficientNetB0** transfer learning (TensorFlow/Keras), optional **YOLOv8** detection on the bundled dataset, and a **Streamlit** app.

## Dataset layout (classification)

By default, **`data/classification_dataset/`** is **created automatically** from **`object_detection_Dataset/`**: images are copied into `TRAIN` / `VALID` / `TEST` with `bird` / `drone` subfolders using the **majority YOLO class** in each image’s `.txt` label (class `0` → bird, `1` → drone). This runs when you start training or evaluation, or you can run:

`py -3 scripts/build_classification_from_yolo.py`

You can still supply your own layout under `data/classification_dataset/`; if it is already complete, it will not be overwritten.

Data_Link: https://drive.google.com/drive/folders/1nn1vqsh8juhafkJcleembrjQ9EqtIoMh?usp=sharing

Download and upload it data folder.

If you use a **custom** classification tree instead, place:

- `TRAIN/bird`, `TRAIN/drone`
- `VALID/bird`, `VALID/drone`
- `TEST/bird`, `TEST/drone`

## Quick commands

From the **repository root**. On Windows, the commands below use the **Python Launcher** (`py`), which matches a typical working setup when `python` / `pip` are not on `PATH`. On Linux or macOS, substitute `python3` (or `python`) and `python3 -m pip` as needed.

**Dependencies** (run once per environment):

```powershell
py -m pip install -r requirements.txt
```

**Scripts and app** (copy-paste as-is on Windows):

```powershell
# EDA: counts + optional sample grid
py scripts/explore_classification_data.py --plot

# Optional: build classification folders from YOLO labels only (otherwise created on first train/eval)
py -3 scripts/build_classification_from_yolo.py

# Train classifiers
py scripts/train_classification.py --model cnn --epochs 40
py scripts/train_classification.py --model efficientnet --epochs 40

# Evaluate on TEST (confusion matrix, curves, JSON metrics)
py scripts/evaluate_classification.py --weights outputs/models/best_cnn.keras
py scripts/evaluate_classification.py --weights outputs/models/best_efficientnet.keras
py scripts/evaluate_classification.py --compare-histories

# Comparison report (after metrics exist)
py scripts/compare_models.py

# YOLOv8 — uses object_detection_Dataset/ (or set AERIAL_OBJECT_DETECTION_ROOT)
py scripts/train_yolo.py --epochs 50
# First-time weights only: if GitHub download fails with SSL/curl errors, use:
#   py scripts/train_yolo.py --epochs 50 --insecure-download
# or set AERIAL_YOLO_INSECURE_DOWNLOAD=1, or place yolov8n.pt under ./weights/

# Streamlit UI
py -3 -m streamlit run app/streamlit_app.py
```

The app has three tabs: **Inference** (YOLO / CNN / EfficientNet + upload), **Run pipelines / combinations** (preset and custom training & evaluation via buttons), and **Final results** (metrics table, JSON, Markdown report, figures). A **Results overview** banner above the tabs shows test accuracy, F1, YOLO `best.pt`, and training times when available; after any pipeline run, the Run tab lists **updated tables and recent figures** read from disk.

The app loads **TensorFlow only in Classification mode** so the home screen appears quickly. If the browser stays blank, try `http://127.0.0.1:8501` or `http://localhost:8501` and watch the **Running…** indicator while TensorFlow loads the first time.

## Object detection data

YOLO data is read from **`object_detection_Dataset/`** at the repo root. It contains `train`, `valid`, `test` with `images` and `labels`. The bundled `data.yaml` omits a top-level `path` key so Ultralytics resolves splits relative to that folder (do not set `path: .` alone, or runs may look under the wrong directory).

To point at another copy of the dataset, set **`AERIAL_OBJECT_DETECTION_ROOT`** to that directory (must contain `data.yaml`, `train`, `valid`, etc.).

## Outputs

- Trained Keras models: `outputs/models/`
- Figures and reports: `outputs/figures/`, `outputs/model_comparison_report.md`
- YOLO runs: `outputs/yolo/`

## Notebooks

See `notebooks/01_classification_eda.ipynb` for a minimal EDA template.

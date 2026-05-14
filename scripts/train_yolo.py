"""Train YOLOv8 on the object_detection_Dataset (Bird / drone)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

from src import config
from src.yolo_weights import resolve_yolo_weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=config.YOLO_DATA_YAML,
        help=f"data.yaml path (default: {config.YOLO_DATA_YAML})",
    )
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Ultralytics starter weights")
    parser.add_argument(
        "--insecure-download",
        action="store_true",
        help="If weights are missing, download from GitHub without SSL verify (fixes some corporate "
        "Windows TLS/revocation errors). Or set AERIAL_YOLO_INSECURE_DOWNLOAD=1.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", type=Path, default=config.YOLO_OUTPUT_DIR)
    parser.add_argument("--name", type=str, default="aerial_bird_drone")
    args = parser.parse_args()

    if not args.data.is_file():
        raise FileNotFoundError(args.data)

    args.project.mkdir(parents=True, exist_ok=True)
    config.ensure_output_dirs()

    weights_path = resolve_yolo_weights(
        args.model,
        config.YOLO_WEIGHTS_DIR,
        insecure_download=args.insecure_download,
    )
    model = YOLO(weights_path)
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
    )
    print("Training finished. Best weights typically at:")
    print(args.project / args.name / "weights" / "best.pt")


if __name__ == "__main__":
    main()

"""Populate data/classification_dataset from object_detection_Dataset (YOLO labels)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config
from src.yolo_to_classification import materialize_from_yolo


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--yolo-root", type=Path, default=config.OBJECT_DETECTION_ROOT)
    p.add_argument("--dest", type=Path, default=config.CLASSIFICATION_ROOT)
    p.add_argument("--no-clear", action="store_true", help="Do not remove existing TRAIN/VALID/TEST first")
    args = p.parse_args()
    out = materialize_from_yolo(args.yolo_root, args.dest, clear=not args.no_clear)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

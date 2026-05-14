"""Build TRAIN/VALID/TEST bird|drone folders from a YOLO-format dataset (images + .txt labels)."""
from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path

from src import config

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# YOLO class index -> Keras ImageFolder name (matches data.yaml order: Bird, drone)
CLASS_ID_TO_FOLDER = {0: "bird", 1: "drone"}

YOLO_SPLIT_TO_DEST = {"train": "TRAIN", "valid": "VALID", "test": "TEST"}


def _majority_class_id(label_path: Path) -> int | None:
    if not label_path.is_file():
        return None
    ids: list[int] = []
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        try:
            ids.append(int(float(parts[0])))
        except ValueError:
            continue
    if not ids:
        return None
    return Counter(ids).most_common(1)[0][0]


def splits_ready(classification_root: Path) -> bool:
    """TRAIN/VALID/TEST each have bird/ and drone/ with at least one image."""
    for split in ("TRAIN", "VALID", "TEST"):
        sd = classification_root / split
        for cls in ("bird", "drone"):
            d = sd / cls
            if not d.is_dir():
                return False
            if not any(p.is_file() and p.suffix.lower() in IMAGE_EXTS for p in d.iterdir()):
                return False
    return True


def materialize_from_yolo(
    yolo_root: Path | None = None,
    dest_root: Path | None = None,
    *,
    clear: bool = True,
) -> dict[str, object]:
    """
    Copy images from yolo_root/{train,valid,test}/images into
    dest_root/{TRAIN,VALID,TEST}/{bird,drone}/ using label majority class.
    """
    yolo_root = (yolo_root or config.OBJECT_DETECTION_ROOT).resolve()
    dest_root = (dest_root or config.CLASSIFICATION_ROOT).resolve()

    if not (yolo_root / "train" / "images").is_dir():
        raise FileNotFoundError(
            f"No YOLO images under {yolo_root / 'train' / 'images'}. "
            "Check OBJECT_DETECTION_ROOT / AERIAL_OBJECT_DETECTION_ROOT."
        )

    if clear:
        for dest_split in YOLO_SPLIT_TO_DEST.values():
            split_dir = dest_root / dest_split
            if split_dir.is_dir():
                shutil.rmtree(split_dir)

    stats: dict[str, dict[str, int]] = {}
    skipped = 0

    for yolo_split, dest_split in YOLO_SPLIT_TO_DEST.items():
        img_dir = yolo_root / yolo_split / "images"
        lbl_dir = yolo_root / yolo_split / "labels"
        if not img_dir.is_dir():
            continue
        stats[dest_split] = {"bird": 0, "drone": 0}
        for cls in ("bird", "drone"):
            (dest_root / dest_split / cls).mkdir(parents=True, exist_ok=True)

        for img_path in sorted(img_dir.iterdir()):
            if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            label_path = lbl_dir / f"{img_path.stem}.txt"
            cid = _majority_class_id(label_path)
            if cid is None or cid not in CLASS_ID_TO_FOLDER:
                skipped += 1
                continue
            folder = CLASS_ID_TO_FOLDER[cid]
            dest = dest_root / dest_split / folder / img_path.name
            shutil.copy2(img_path, dest)
            stats[dest_split][folder] += 1

    return {"dest": str(dest_root), "yolo_source": str(yolo_root), "per_split": stats, "skipped_no_label": skipped}


def ensure_classification_from_yolo(
    classification_root: Path | None = None,
    yolo_root: Path | None = None,
) -> bool:
    """
    If classification folders are not ready, build them from the YOLO dataset.
    Returns True if splits are ready after the call.
    """
    classification_root = (classification_root or config.CLASSIFICATION_ROOT).resolve()
    if splits_ready(classification_root):
        return True
    print(
        "Classification folders missing or incomplete; building from YOLO dataset "
        f"({config.OBJECT_DETECTION_ROOT}) …",
        flush=True,
    )
    materialize_from_yolo(yolo_root=yolo_root, dest_root=classification_root, clear=True)
    if not splits_ready(classification_root):
        raise RuntimeError(
            "Could not build a valid classification layout under "
            f"{classification_root}. Check YOLO images/labels and class ids 0–1."
        )
    return True

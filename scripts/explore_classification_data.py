"""Count images per split/class and optionally show a small montage."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config


def count_split(split_dir: Path) -> dict[str, int]:
    if not split_dir.is_dir():
        return {}
    counts: dict[str, int] = {}
    for cls in sorted(p.name for p in split_dir.iterdir() if p.is_dir()):
        n = sum(1 for p in (split_dir / cls).rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        counts[cls] = n
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect classification_dataset folder counts.")
    parser.add_argument(
        "--root",
        type=Path,
        default=config.CLASSIFICATION_ROOT,
        help="Root folder containing TRAIN, VALID, TEST",
    )
    parser.add_argument("--plot", action="store_true", help="Save a 2x4 grid of random samples to outputs/figures")
    args = parser.parse_args()

    if args.root.resolve() == config.CLASSIFICATION_ROOT.resolve():
        from src.yolo_to_classification import ensure_classification_from_yolo

        ensure_classification_from_yolo()

    root: Path = args.root
    if not root.is_dir():
        print(f"Missing dataset root: {root}")
        print("Place TRAIN/VALID/TEST with bird/ and drone/ subfolders under data/classification_dataset/")
        raise SystemExit(1)

    for name in ("TRAIN", "VALID", "TEST"):
        d = root / name
        counts = count_split(d)
        total = sum(counts.values())
        print(f"{name}: {counts}  (total={total})")

    if args.plot:
        import random

        import matplotlib.pyplot as plt
        from PIL import Image

        config.ensure_output_dirs()
        train_bird = list((root / "TRAIN" / "bird").glob("*.jpg"))[:200]
        train_drone = list((root / "TRAIN" / "drone").glob("*.jpg"))[:200]
        if not train_bird or not train_drone:
            print("Not enough images to plot.")
            return
        fig, axes = plt.subplots(2, 4, figsize=(10, 5))
        for row, (paths, title) in enumerate([(train_bird, "bird"), (train_drone, "drone")]):
            for col in range(4):
                p = random.choice(paths)
                ax = axes[row, col]
                ax.imshow(Image.open(p).convert("RGB"))
                ax.axis("off")
                if col == 0:
                    ax.set_ylabel(title)
        plt.tight_layout()
        out = config.FIGURES_DIR / "sample_grid.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()

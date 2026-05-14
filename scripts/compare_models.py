"""Build a short model comparison report from saved training timings and test metrics."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config


def _load_json(p: Path) -> dict | None:
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    config.ensure_output_dirs()
    out = ROOT / "outputs" / "model_comparison_report.md"

    cnn_t = _load_json(config.MODELS_DIR / "timing_cnn.json")
    eff_t = _load_json(config.MODELS_DIR / "timing_efficientnet.json")
    cnn_m = _load_json(config.MODELS_DIR / "test_metrics_best_cnn.json")
    eff_m = _load_json(config.MODELS_DIR / "test_metrics_best_efficientnet.json")

    lines = [
        "# Model comparison (Bird vs Drone classification)",
        "",
        "Generated from `outputs/models/timing_*.json` and `test_metrics_*.json`.",
        "Run `train_classification.py` for both models, then `evaluate_classification.py` for each checkpoint.",
        "",
        "## Training time",
        "",
        "| Model | Train (s) | Epochs |",
        "| --- | ---: | ---: |",
    ]
    if cnn_t:
        lines.append(f"| Custom CNN | {cnn_t.get('train_seconds', '—')} | {cnn_t.get('epochs_ran', '—')} |")
    if eff_t:
        lines.append(f"| EfficientNetB0 (TL) | {eff_t.get('train_seconds', '—')} | {eff_t.get('epochs_ran', '—')} |")
    if not cnn_t and not eff_t:
        lines.append("| (no timing files yet) | | |")

    lines.extend(
        [
            "",
            "## Test set",
            "",
            "| Model | Accuracy | F1 (drone positive) |",
            "| --- | ---: | ---: |",
        ]
    )
    if cnn_m:
        lines.append(f"| Custom CNN | {cnn_m.get('accuracy', 0):.4f} | {cnn_m.get('f1_drone_positive', 0):.4f} |")
    if eff_m:
        lines.append(f"| EfficientNetB0 (TL) | {eff_m.get('accuracy', 0):.4f} | {eff_m.get('f1_drone_positive', 0):.4f} |")
    if not cnn_m and not eff_m:
        lines.append("| (no test metric files yet) | | |")

    lines.extend(["", "## Generalization", "", "Prefer higher test accuracy and F1 with similar or lower validation–test gap (inspect confusion matrices in `outputs/figures/`).", ""])

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

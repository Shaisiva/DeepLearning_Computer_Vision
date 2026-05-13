"""Streamlit UI: inference, pipeline combinations, and final results dashboard."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Aerial Bird vs Drone", layout="wide")

from src import config

import numpy as np
from PIL import Image

SCRIPTS = ROOT / "scripts"
PY = Path(sys.executable)

CNN_KERAS = config.MODELS_DIR / "best_cnn.keras"
EFF_KERAS = config.MODELS_DIR / "best_efficientnet.keras"

# --- job registry (id -> label) ---
JOB_LABELS: dict[str, str] = {
    "build_class_from_yolo": "Build TRAIN/VALID/TEST from YOLO labels (copy images)",
    "train_cnn": "Train Custom CNN",
    "train_efficientnet": "Train EfficientNetB0 (transfer)",
    "train_yolo": "Train YOLOv8",
    "eval_cnn": "Evaluate Custom CNN (test set)",
    "eval_efficientnet": "Evaluate EfficientNet (test set)",
    "compare_histories": "Plot CNN vs EfficientNet val curves",
    "compare_report": "Write model comparison report (Markdown)",
    "explore_data": "Count images per split (classification dataset)",
}

FULL_PIPELINE: list[str] = [
    "build_class_from_yolo",
    "train_cnn",
    "train_efficientnet",
    "train_yolo",
    "eval_cnn",
    "eval_efficientnet",
    "compare_histories",
    "compare_report",
]


def _script_argv(name: str) -> list[str]:
    return [str(PY), str(SCRIPTS / name)]


def build_job_command(job_id: str, clf_epochs: int, yolo_epochs: int, yolo_batch: int) -> list[str] | None:
    if job_id == "build_class_from_yolo":
        return _script_argv("build_classification_from_yolo.py")
    if job_id == "train_cnn":
        return _script_argv("train_classification.py") + ["--model", "cnn", "--epochs", str(clf_epochs)]
    if job_id == "train_efficientnet":
        return _script_argv("train_classification.py") + ["--model", "efficientnet", "--epochs", str(clf_epochs)]
    if job_id == "train_yolo":
        return _script_argv("train_yolo.py") + ["--epochs", str(yolo_epochs), "--batch", str(yolo_batch)]
    if job_id == "eval_cnn":
        return _script_argv("evaluate_classification.py") + ["--weights", str(CNN_KERAS)]
    if job_id == "eval_efficientnet":
        return _script_argv("evaluate_classification.py") + ["--weights", str(EFF_KERAS)]
    if job_id == "compare_histories":
        return _script_argv("evaluate_classification.py") + ["--compare-histories"]
    if job_id == "compare_report":
        return _script_argv("compare_models.py")
    if job_id == "explore_data":
        return _script_argv("explore_classification_data.py")
    return None


def run_subprocess(argv: list[str]) -> tuple[int, str]:
    env = os.environ.copy()
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    env.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    proc = subprocess.run(
        argv,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=None,
        env=env,
    )
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    if err:
        out = (out + "\n\n--- stderr ---\n" + err).strip()
    return proc.returncode, out


def clear_model_caches() -> None:
    load_keras_model.clear()
    load_yolo.clear()


@st.cache_resource
def load_keras_model(path_str: str, mtime_ns: int):
    keras = _import_keras()
    return keras.models.load_model(Path(path_str), compile=False)


def _import_keras():
    from tensorflow import keras

    return keras


@st.cache_resource
def load_yolo(weights_path: str):
    from ultralytics import YOLO

    return YOLO(weights_path)


def predict_classification(model, image: Image.Image) -> tuple[str, float]:
    keras = _import_keras()
    img = image.convert("RGB").resize(config.IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32)
    batch = np.expand_dims(arr, 0)
    p_drone = float(model.predict(batch, verbose=0)[0, 0])
    label = "drone" if p_drone >= 0.5 else "bird"
    conf = p_drone if label == "drone" else 1.0 - p_drone
    return label, conf


def gather_test_metrics_rows() -> list[dict]:
    """Rows for dataframe / metrics from outputs/models/test_metrics_*.json."""
    rows: list[dict] = []
    for p in sorted(config.MODELS_DIR.glob("test_metrics_*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            name = p.name.replace("test_metrics_", "").replace(".json", "")
            acc = float(d.get("accuracy", 0))
            f1 = float(d.get("f1_drone_positive", 0))
            rows.append(
                {
                    "model": name.replace("_", " ").title(),
                    "accuracy": acc,
                    "accuracy_pct": acc * 100.0 if acc <= 1.0 else acc,
                    "f1_drone_positive": f1,
                }
            )
        except (json.JSONDecodeError, TypeError, ValueError, OSError):
            continue
    return rows


def gather_timing_rows() -> list[dict]:
    rows: list[dict] = []
    for p in sorted(config.MODELS_DIR.glob("timing_*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            name = p.stem.replace("timing_", "")
            rows.append(
                {
                    "model": name,
                    "train_s": d.get("train_seconds"),
                    "epochs": d.get("epochs_ran"),
                }
            )
        except (json.JSONDecodeError, TypeError, OSError):
            continue
    return rows


def render_results_banner() -> None:
    """Always-visible strip: test accuracy / F1 and YOLO checkpoint."""
    rows = gather_test_metrics_rows()
    yolo_weights = sorted(
        config.YOLO_OUTPUT_DIR.glob("**/weights/best.pt"),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    if not rows and not yolo_weights:
        st.info(
            "**No saved results yet.** Run jobs under **Run pipelines / combinations** "
            "(e.g. *Evaluate both + report* or *FULL pipeline*), then metrics and figures appear here."
        )
        return

    st.markdown("#### Results overview")
    if rows:
        cols = st.columns(len(rows))
        for i, r in enumerate(rows):
            with cols[i]:
                pct = r["accuracy_pct"]
                st.metric(
                    r["model"],
                    f"{pct:.2f}%",
                    f"F1 (drone) {r['f1_drone_positive']:.4f}",
                    help="Test-set accuracy and F1 from saved JSON",
                )
    if yolo_weights:
        st.success(f"**YOLOv8 best weights:** `{yolo_weights[0]}`")

    timing = gather_timing_rows()
    if timing:
        parts = [f"**{t['model']}** train: {t.get('train_s', '—')}s ({t.get('epochs', '—')} epochs)" for t in timing]
        st.caption(" · ".join(parts))


def render_results_compact(title: str = "Saved outputs (disk)") -> None:
    """Metrics table + recent figures — shown after pipeline runs and on demand."""
    st.subheader(title)
    rows = gather_test_metrics_rows()
    if rows:
        st.dataframe(
            [
                {
                    "Model": r["model"],
                    "Test accuracy": round(r["accuracy"], 4),
                    "F1 (drone +)": round(r["f1_drone_positive"], 4),
                }
                for r in rows
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("No `test_metrics_*.json` files yet.")

    figs = sorted(config.FIGURES_DIR.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)[:8]
    if figs:
        st.markdown("**Recent figures**")
        fc = st.columns(min(4, len(figs)))
        for i, fp in enumerate(figs):
            with fc[i % len(fc)]:
                st.image(str(fp), caption=fp.name, use_container_width=True)

    report = ROOT / "outputs" / "model_comparison_report.md"
    if report.is_file():
        with st.expander("Model comparison report (Markdown)", expanded=False):
            st.markdown(report.read_text(encoding="utf-8"))


def run_job_sequence(job_ids: list[str], clf_epochs: int, yolo_epochs: int, yolo_batch: int) -> list[dict]:
    config.ensure_output_dirs()
    results: list[dict] = []
    for jid in job_ids:
        cmd = build_job_command(jid, clf_epochs, yolo_epochs, yolo_batch)
        if cmd is None:
            results.append({"job": jid, "ok": False, "log": "Unknown job id"})
            continue
        code, log = run_subprocess(cmd)
        results.append({"job": jid, "label": JOB_LABELS.get(jid, jid), "ok": code == 0, "exit_code": code, "log": log})
        if jid.startswith("train_"):
            clear_model_caches()
    return results


def render_results_dashboard() -> None:
    rows = gather_test_metrics_rows()
    timing_files = sorted(config.MODELS_DIR.glob("timing_*.json"))

    st.subheader("Summary (test metrics)")
    if rows:
        st.dataframe(
            [
                {
                    "Model": r["model"],
                    "Test accuracy": round(r["accuracy"], 4),
                    "F1 (drone +)": round(r["f1_drone_positive"], 4),
                }
                for r in rows
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No `test_metrics_*.json` yet. Run evaluation jobs from **Run pipelines**.")

    metrics_files = sorted(config.MODELS_DIR.glob("test_metrics_*.json"))
    st.subheader("Saved metrics (JSON)")
    if not metrics_files and not timing_files:
        st.caption("No timing files either.")
    for p in metrics_files + timing_files:
        with st.expander(f"`{p.name}`"):
            try:
                st.json(json.loads(p.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                st.text(p.read_text(encoding="utf-8"))

    st.subheader("Model comparison report")
    report = ROOT / "outputs" / "model_comparison_report.md"
    if report.is_file():
        st.markdown(report.read_text(encoding="utf-8"))
    else:
        st.info("Run **Write model comparison report** (after metrics exist) to create `outputs/model_comparison_report.md`.")

    st.subheader("Figures")
    figs = sorted(config.FIGURES_DIR.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not figs:
        st.info("No PNG figures in `outputs/figures/` yet.")
    else:
        cols = st.columns(min(3, len(figs)))
        for i, fp in enumerate(figs):
            with cols[i % len(cols)]:
                st.image(str(fp), caption=fp.name, use_container_width=True)

    st.subheader("Latest YOLO run (if any)")
    yolo_weights = sorted(config.YOLO_OUTPUT_DIR.glob("**/weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if yolo_weights:
        st.success(f"Best weights: `{yolo_weights[0]}`")
    else:
        st.caption("No `best.pt` under `outputs/yolo/` yet.")


# --- main UI ---
st.title("Aerial object classification and detection")
st.caption("Bird vs Drone — train, evaluate, compare, and run inference from one place.")

render_results_banner()
st.divider()

tab_inf, tab_run, tab_res = st.tabs(["Inference", "Run pipelines / combinations", "Final results"])

with st.sidebar:
    st.markdown("**Tips**")
    st.markdown(
        "- **Inference → TensorFlow** loads only for Keras models (can take 1–2 min first time).\n"
        "- **Run pipelines** uses the same Python as Streamlit (`sys.executable`).\n"
        "- Training can take a long time; lower epochs for a quick test.\n"
        "- Try [http://127.0.0.1:8501](http://127.0.0.1:8501) if the page does not load."
    )

# ----- Inference -----
with tab_inf:
    st.markdown("### Choose model and upload an image")
    model_choice = st.radio(
        "Model",
        ["YOLOv8 (detection)", "Custom CNN (classification)", "EfficientNetB0 (classification)"],
        horizontal=True,
    )

    up = st.file_uploader("Image", type=["jpg", "jpeg", "png"], key="inf_upload")

    if up is not None:
        image = Image.open(up)
        st.image(image, caption="Input", use_container_width=True)

        if model_choice == "YOLOv8 (detection)":
            yolo_weights = list(config.YOLO_OUTPUT_DIR.glob("**/weights/best.pt"))
            if not yolo_weights:
                st.error("No YOLO `best.pt` found. Train from **Run pipelines** tab.")
            else:
                weights = sorted(yolo_weights, key=lambda p: p.stat().st_mtime, reverse=True)[0]
                with st.spinner(f"Loading YOLO ({weights.name})…"):
                    model = load_yolo(str(weights))
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    image.convert("RGB").save(tmp.name, quality=95)
                    with st.spinner("Detecting…"):
                        res = model.predict(tmp.name, verbose=False)[0]
                out_arr = res.plot()
                if out_arr is not None and out_arr.ndim == 3 and out_arr.shape[-1] == 3:
                    out_arr = out_arr[..., ::-1]
                st.image(Image.fromarray(out_arr), caption="YOLO output (boxes)", use_container_width=True)
                try:
                    names = getattr(res, "names", None)
                    txt = []
                    for b in res.boxes:
                        cls_id = int(b.cls[0])
                        conf = float(b.conf[0])
                        label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
                        txt.append(f"- {label}: **{conf:.3f}**")
                    if txt:
                        st.markdown("**Detections:**\n" + "\n".join(txt))
                    else:
                        st.info("No boxes above threshold.")
                except Exception:
                    st.caption("Could not parse box list.")

        elif model_choice == "Custom CNN (classification)":
            if not CNN_KERAS.is_file():
                st.error(f"Missing `{CNN_KERAS.name}`. Train **Custom CNN** from the Run tab.")
            else:
                with st.spinner("Loading TensorFlow + CNN…"):
                    m = load_keras_model(str(CNN_KERAS), CNN_KERAS.stat().st_mtime_ns)
                with st.spinner("Predicting…"):
                    label, conf = predict_classification(m, image)
                st.success(f"**{label}** — confidence **{conf:.3f}** (CNN)")

        else:
            if not EFF_KERAS.is_file():
                st.error(f"Missing `{EFF_KERAS.name}`. Train **EfficientNet** from the Run tab.")
            else:
                with st.spinner("Loading TensorFlow + EfficientNet…"):
                    m = load_keras_model(str(EFF_KERAS), EFF_KERAS.stat().st_mtime_ns)
                with st.spinner("Predicting…"):
                    label, conf = predict_classification(m, image)
                st.success(f"**{label}** — confidence **{conf:.3f}** (EfficientNetB0)")

    with st.expander("Saved test metrics (from disk)", expanded=False):
        rrows = gather_test_metrics_rows()
        if rrows:
            st.dataframe(
                [
                    {"Model": r["model"], "Accuracy": r["accuracy"], "F1 drone+": r["f1_drone_positive"]}
                    for r in rrows
                ],
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.caption("Run evaluations on the **Run pipelines** tab to create `test_metrics_*.json`.")

# ----- Run pipelines -----
with tab_run:
    st.markdown("### Parameters (used by training jobs)")
    c1, c2, c3 = st.columns(3)
    with c1:
        clf_epochs = st.number_input("Classification epochs", min_value=1, max_value=200, value=5, step=1)
    with c2:
        yolo_epochs = st.number_input("YOLO epochs", min_value=1, max_value=300, value=3, step=1)
    with c3:
        yolo_batch = st.number_input("YOLO batch size", min_value=1, max_value=128, value=8, step=1)

    st.markdown("### Preset combinations (one click)")
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        if st.button("Train both classifiers", help="CNN then EfficientNet"):
            st.session_state["_last_run"] = run_job_sequence(["train_cnn", "train_efficientnet"], clf_epochs, yolo_epochs, yolo_batch)
    with p2:
        if st.button("Train YOLO only"):
            st.session_state["_last_run"] = run_job_sequence(["train_yolo"], clf_epochs, yolo_epochs, yolo_batch)
    with p3:
        if st.button("Evaluate both + report", help="Eval CNN, Eval EfficientNet, val curves, Markdown report"):
            st.session_state["_last_run"] = run_job_sequence(
                ["eval_cnn", "eval_efficientnet", "compare_histories", "compare_report"],
                clf_epochs,
                yolo_epochs,
                yolo_batch,
            )
    with p4:
        if st.button("FULL pipeline", help="All train + eval + compare (long run)"):
            st.session_state["_last_run"] = run_job_sequence(FULL_PIPELINE, clf_epochs, yolo_epochs, yolo_batch)

    st.markdown("### Individual jobs")
    ids = list(JOB_LABELS.keys())
    cols = st.columns(4)
    for i, jid in enumerate(ids):
        with cols[i % 4]:
            if st.button(JOB_LABELS[jid], key=f"btn_{jid}"):
                st.session_state["_last_run"] = run_job_sequence([jid], clf_epochs, yolo_epochs, yolo_batch)

    st.markdown("### Custom sequence")
    chosen = st.multiselect("Select jobs (order preserved)", options=ids, format_func=lambda x: JOB_LABELS[x])
    if st.button("Run custom sequence", type="primary"):
        if not chosen:
            st.warning("Select at least one job.")
        else:
            st.session_state["_last_run"] = run_job_sequence(chosen, clf_epochs, yolo_epochs, yolo_batch)

    st.markdown("---")
    st.subheader("Latest run output")
    last = st.session_state.get("_last_run")
    if last is None:
        st.caption("Run a preset, a single job, or a custom sequence to see logs here.")
    else:
        ok_all = all(r.get("ok") for r in last)
        if ok_all:
            st.success("All steps finished with exit code 0.")
        else:
            st.error("Some steps failed. Expand logs below.")
        for r in last:
            label = r.get("label", r.get("job", "?"))
            status = "OK" if r.get("ok") else f"FAIL (exit {r.get('exit_code')})"
            with st.expander(f"{status} — {label}", expanded=not r.get("ok")):
                log = r.get("log", "")
                st.code(log[-12000:] if len(log) > 12000 else log or "(no output)", language="text")

    if last is not None:
        render_results_compact("Outputs after last pipeline run (read from disk)")

# ----- Final results -----
with tab_res:
    if st.button("Refresh from disk", key="refresh_results"):
        st.rerun()
    render_results_dashboard()

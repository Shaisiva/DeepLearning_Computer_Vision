"""Resolve or fetch Ultralytics pretrained .pt weights without relying on curl (Windows SSL/revocation issues)."""
from __future__ import annotations

import os
import ssl
import urllib.error
import urllib.request
from pathlib import Path

# Matches ultralytics.utils.downloads.attempt_download_asset default for ultralytics/assets.
_GITHUB_ASSETS_BASE = "https://github.com/ultralytics/assets/releases/download/v8.4.0"

_INSECURE_ENV = "AERIAL_YOLO_INSECURE_DOWNLOAD"


def _ultralytics_weights_dir() -> Path | None:
    try:
        from ultralytics.utils import SETTINGS

        return Path(SETTINGS["weights_dir"])
    except Exception:
        return None


def _candidate_paths(filename: str, repo_weights: Path) -> list[Path]:
    name = Path(filename).name
    out: list[Path] = [repo_weights / name, Path.cwd() / name]
    ud = _ultralytics_weights_dir()
    if ud is not None:
        out.append(ud / name)
    return out


def _download_github_asset(filename: str, dest: Path, *, verify_ssl: bool) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{_GITHUB_ASSETS_BASE}/{Path(filename).name}"
    context = None if verify_ssl else ssl._create_unverified_context()
    req = urllib.request.Request(url, headers={"User-Agent": "aerial-cv-yolo-bootstrap"})
    with urllib.request.urlopen(req, context=context, timeout=600) as resp:
        data = resp.read()
    dest.write_bytes(data)
    if dest.stat().st_size < 100_000:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"Download from {url} produced a tiny file; check URL or network.")
    return dest


def resolve_yolo_weights(
    model: str,
    repo_weights: Path,
    *,
    insecure_download: bool,
) -> str:
    """
    Return an absolute path or URI string suitable for YOLO().

    If ``model`` is a bare asset name (e.g. yolov8n.pt), looks under ``repo_weights``,
    the current working directory, then Ultralytics' configured weights_dir. If still
    missing, downloads from GitHub releases when ``insecure_download`` is True or when
    env ``AERIAL_YOLO_INSECURE_DOWNLOAD`` is 1/true/yes (SSL verification disabled — use
    only on broken corporate TLS).
    """
    raw = model.strip().strip("'\"")
    p = Path(raw)

    if p.is_file():
        return str(p.resolve())

    if "://" in raw:
        return raw

    name = p.name
    if not name.endswith(".pt") and "/" not in raw.replace("\\", "/"):
        return raw

    for c in _candidate_paths(name, repo_weights):
        if c.is_file():
            return str(c.resolve())

    allow_insecure = insecure_download or os.environ.get(_INSECURE_ENV, "").lower() in (
        "1",
        "true",
        "yes",
    )
    dest = (repo_weights / name).resolve()

    # Try strict SSL first when user did not opt into insecure (helps if only curl was broken).
    if not allow_insecure:
        try:
            _download_github_asset(name, dest, verify_ssl=True)
            return str(dest)
        except (urllib.error.URLError, ssl.SSLError, OSError, RuntimeError):
            dest.unlink(missing_ok=True)

        raise FileNotFoundError(
            f"Could not find or download '{name}'. Place the file under {repo_weights} or the "
            f"repo root, or retry with --insecure-download (or set {_INSECURE_ENV}=1) if SSL/curl "
            "errors block downloads on your network."
        ) from None

    _download_github_asset(name, dest, verify_ssl=False)
    return str(dest)

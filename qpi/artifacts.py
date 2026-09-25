"""Versioned artifacts; completed runs are immutable and partial runs explicit."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import numpy as np
from .config import Config
from .physics import prescription, sampling_warnings


def json_write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance():
    root = Path(__file__).resolve().parents[1]
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unavailable"
    versions = {}
    for name in ("jax", "jaxlib", "numpy", "optax", "chromatix", "scipy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    import jax
    sources = sorted((root/"qpi").glob("*.py")) + sorted((root/"chromatix"/"src").rglob("*.py")) + [root/"oas_core.py"]
    return {"git_commit": git("rev-parse", "HEAD"), "git_status": git("status", "--porcelain"),
            "source_sha256": {str(p.relative_to(root)): file_hash(p) for p in sources},
            "python": platform.python_version(), "platform": platform.platform(),
            "packages": versions, "devices": [str(d) for d in jax.devices()]}


def save_result(root, config, result, manifest_hash, source):
    root = Path(root)
    np.savez_compressed(root / "parameters.npz", **result["parameters"])
    summary = {k: v for k, v in result.items() if k != "parameters"}
    summary["architecture"] = result["architecture"].tolist()
    summary.update(schema=1, config=config.to_dict(), dataset_manifest_sha256=manifest_hash,
                   parameters_sha256=file_hash(root / "parameters.npz"), provenance=source,
                   prescription=prescription(result["parameters"], result["architecture"], config),
                   sampling_warnings=sampling_warnings(result["parameters"], result["architecture"], config))
    json_write(root / "result.json", summary)
    return summary


def load_result(root):
    root = Path(root)
    summary = json.loads((root / "result.json").read_text())
    if summary["schema"] != 1 or file_hash(root / "parameters.npz") != summary["parameters_sha256"]:
        raise ValueError("Unknown schema or corrupted parameter checkpoint")
    with np.load(root / "parameters.npz", allow_pickle=False) as f:
        parameters = {k: f[k] for k in f.files}
    c = Config(**summary["config"])
    return c, summary, parameters

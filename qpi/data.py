"""Offline phase phantoms and immutable dataset manifests; phases are radians."""
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np

SPLITS = {"train": 0, "validation": 1, "test": 2, "ood": 3}


def digest(array):
    a = np.ascontiguousarray(array)
    return hashlib.sha256(str(a.shape).encode() + a.dtype.str.encode() + a.tobytes()).hexdigest()


def roi(size):
    # Fixed in advance; never inferred from test truth or signal intensity.
    axis = (np.arange(size) - size // 2) / (size / 2)
    y, x = np.meshgrid(axis, axis, indexing="ij")
    return (np.abs(x) <= .75) & (np.abs(y) <= .75)


def phantom(size, seed, amplitude, ood=False):
    rng = np.random.default_rng(seed)
    axis = (np.arange(size) - size // 2) / (size / 2)
    y, x = np.meshgrid(axis, axis, indexing="ij")
    image = np.zeros((size, size))
    for _ in range(rng.integers(2, 6)):
        cx, cy = rng.uniform(-.35, .35, 2)
        sx, sy = rng.uniform(.10, .25, 2)
        angle = rng.uniform(-np.pi, np.pi)
        u = (x-cx)*np.cos(angle) + (y-cy)*np.sin(angle)
        v = -(x-cx)*np.sin(angle) + (y-cy)*np.cos(angle)
        r2 = (u/sx)**2 + (v/sy)**2
        bump = (r2 < 1).astype(float) if ood else np.exp(-r2/2)
        image += rng.uniform(.3, 1) * bump
    # Known zero-phase background fixes the phase convention. Smooth support.
    support = np.clip((.7 - np.maximum(np.abs(x), np.abs(y))) / .15, 0, 1)
    image *= support**2 * (3 - 2 * support)
    image *= amplitude * rng.uniform(.4, 1) / max(image.max(), 1e-12)
    return image.astype(np.float32)


def generate(config, destination):
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": 1, "kind": "synthetic_phase_phantoms", "units": "radians",
                "phase_convention": "zero in known outer background", "config": config.to_dict(),
                "splits": {}}
    for split, code in SPLITS.items():
        n = getattr(config, "n_" + split) if split != "ood" else config.n_test
        ids = [f"{config.data_seed}:{code}:{i}" for i in range(n)]
        phases = np.stack([phantom(config.size,
            np.random.SeedSequence([config.data_seed, code, i]), config.phase_max_rad,
            ood=(split == "ood")) for i in range(n)])
        np.savez_compressed(root / f"{split}.npz", phase=phases, ids=np.array(ids))
        manifest["splits"][split] = {"count": n, "sha256": digest(phases), "ids": ids,
                                     "specimen_sha256": [digest(x) for x in phases], "group_ids": ids}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def import_directory(config, source, destination):
    """Register pre-split NPZ files. Split at specimen/acquisition-group level."""
    source, root = Path(source), Path(destination)
    root.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": 1, "kind": "user_supplied_phase_maps", "units": "radians",
                "phase_convention": "zero in known outer background", "config": config.to_dict(), "splits": {}}
    for split in SPLITS:
        with np.load(source/f"{split}.npz", allow_pickle=False) as f:
            phases = f["phase"]
            ids, groups = f["ids"].tolist(), f["group_ids"].tolist()
        if not all(isinstance(x, str) for x in ids+groups) or len(groups) != len(ids):
            raise ValueError("ids and group_ids must be one-dimensional string arrays of matching length")
        shutil.copy2(source/f"{split}.npz", root/f"{split}.npz")
        manifest["splits"][split] = {"count": len(phases), "sha256": digest(phases), "ids": ids,
                                     "specimen_sha256": [digest(x) for x in phases], "group_ids": groups}
    (root/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    for split in SPLITS:
        load_split(root, split, config)
    return manifest


def load_split(root, split, config):
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("units") != "radians":
        raise ValueError("Dataset phase units must be radians")
    for key in ("size", "pitch_um", "wavelength_um"):
        if manifest["config"][key] != getattr(config, key):
            raise ValueError(f"Dataset {key} does not match simulation")
    seen_ids, seen_hashes, seen_groups = set(), set(), set()
    for name in SPLITS:
        entry = manifest["splits"][name]
        ids = entry["ids"]
        hashes = entry["specimen_sha256"]
        groups = entry["group_ids"]
        if (len(set(ids)) != len(ids) or seen_ids.intersection(ids)
            or seen_hashes.intersection(hashes) or seen_groups.intersection(groups)):
            raise ValueError("Dataset splits overlap by identifier, specimen hash, or acquisition group")
        seen_ids.update(ids)
        seen_hashes.update(hashes)
        seen_groups.update(groups)
    with np.load(root / f"{split}.npz", allow_pickle=False) as f:
        phase, ids = f["phase"], f["ids"].tolist()
        groups = f["group_ids"].tolist() if "group_ids" in f else ids
    entry = manifest["splits"][split]
    if phase.shape != (entry["count"], config.size, config.size) or entry["count"] < 1:
        raise ValueError("Dataset dimensions do not match configuration/manifest")
    if (digest(phase) != entry["sha256"] or ids != entry["ids"]
        or [digest(x) for x in phase] != entry["specimen_sha256"] or len(ids) != len(phase)
        or groups != entry["group_ids"]):
        raise ValueError("Dataset checksum/identifiers do not match frozen manifest")
    if not np.isfinite(phase).all() or np.max(np.abs(phase)) >= np.pi:
        raise ValueError("Expected finite relative phases with magnitude below pi")
    if np.max(np.abs(phase[:, ~roi(config.size)])) > 1e-6:
        raise ValueError("Known background outside the ROI must have zero phase")
    return phase.astype(np.float32), ids, manifest

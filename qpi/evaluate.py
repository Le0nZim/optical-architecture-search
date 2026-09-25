"""Frozen-design evaluation, exact camera sampling, and numerical convergence."""
from dataclasses import replace
import csv
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from scipy.ndimage import map_coordinates
from .artifacts import file_hash, json_write, load_result, provenance
from .data import SPLITS, load_split, roi
from .physics import intensity, expected_mse, sample_camera, orient


def evaluate_run(run, dataset, output):
    c, result, parameters = load_result(run)
    if file_hash(Path(dataset)/"manifest.json") != result["dataset_manifest_sha256"]:
        raise ValueError("Evaluation dataset differs from training's frozen manifest")
    architecture = jnp.array(result["architecture"], dtype=jnp.int32)
    calibration = result["calibration"]
    forward = jax.jit(jax.vmap(lambda x, dz: intensity(parameters, architecture, x, c, dz),
                              in_axes=(0, None)))
    rows = []
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for split in ("test", "ood"):
        phases, ids, manifest = load_split(dataset, split, c)
        for iz, dz in enumerate(c.evaluation_defocus_um):
            signals = np.asarray(forward(phases, dz))
            throughput = signals.mean(axis=(-2, -1))
            for ip, photons in enumerate(c.evaluation_photons):
                analytic = np.asarray(expected_mse(signals, phases, calibration, c, photons))
                sample_errors, sample_bias = [], []
                first_prediction = None
                for repeat in range(c.noise_repeats):
                    # Stable specimen-specific random streams. Identical seeds
                    # across methods; Poisson samples depend on each rate.
                    measured = np.stack([sample_camera(signal, photons, c,
                        np.random.default_rng(np.random.SeedSequence(
                            [c.data_seed, 9821, SPLITS[split], iz, ip, repeat, i])))
                        for i, signal in enumerate(signals)])
                    measured = np.asarray(orient(measured, calibration["inverted"]))
                    prediction = calibration["gain"]*measured/(photons*c.quantum_efficiency) + calibration["offset"]
                    error = prediction-phases
                    sample_errors.append(np.mean(error[:, roi(c.size)]**2, axis=-1))
                    sample_bias.append(np.mean(error[:, roi(c.size)], axis=-1))
                    if repeat == 0:
                        first_prediction = prediction
                empirical = np.mean(sample_errors, axis=0)
                bias = np.mean(sample_bias, axis=0)
                for i, specimen in enumerate(ids):
                    rows.append({"method": result["method"], "seed": result["seed"],
                        "dataset_kind": manifest["kind"],
                        "split": split, "specimen_id": specimen, "photons_per_input_pixel": photons,
                        "incident_photons_total": photons*c.size**2,
                        "defocus_um": dz, "expected_mse_rad2": float(analytic[i]),
                        "sampled_mse_rad2": float(empirical[i]), "sampled_bias_rad": float(bias[i]),
                        "sensor_throughput": float(throughput[i]),
                        "expected_detected_electrons": float(photons*c.size**2*c.quantum_efficiency*throughput[i]),
                        "components": result["prescription"]["component_count"],
                        "noise_repeats": c.noise_repeats})
                if dz == 0 and photons == c.photons:
                    np.savez_compressed(output/f"examples_{split}.npz", phase_rad=phases[:4],
                        intensity_relative=signals[:4], prediction_rad=first_prediction[:4])
    with (output/"metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    json_write(output/"evaluation.json", {"schema": 1,
        "result_sha256": file_hash(Path(run)/"result.json"),
        "metrics_sha256": file_hash(output/"metrics.csv"),
        "dataset_manifest_sha256": result["dataset_manifest_sha256"],
        "provenance": provenance(), "calibration": calibration,
        "uncertainty_unit": "specimen and independent search seed; never pixels"})
    return rows


def grid_check(run, dataset, destination, count=4):
    c, result, p = load_result(run)
    if file_hash(Path(dataset)/"manifest.json") != result["dataset_manifest_sha256"]:
        raise ValueError("Dataset manifest changed")
    phases, _, _ = load_split(dataset, "validation", c)
    phases = phases[:count]
    arch = jnp.array(result["architecture"])
    def forward(config, batch):
        return np.asarray(jax.jit(jax.vmap(lambda x: intensity(p, arch, x, config)))(batch))
    base = forward(c, phases)
    more_pad = forward(replace(c, padding=max(1, c.padding)*2), phases)
    # Interpolate the object at half the pitch; compare co-located point samples.
    # No optical parameters, masks or readout coefficients are reoptimized.
    fine_axis = np.arange(2*c.size)/2
    coords = np.meshgrid(fine_axis, fine_axis, indexing="ij")
    fine_phases = np.stack([map_coordinates(x, coords, order=1, mode="constant", cval=0.) for x in phases])
    fine = forward(replace(c, size=2*c.size, pitch_um=c.pitch_um/2, padding=2*c.padding), fine_phases)[:, ::2, ::2]
    def compare(other):
        relative = np.sqrt(np.mean((other-base)**2)/max(np.mean(base**2), 1e-20))
        rmse = np.sqrt(np.mean(np.asarray(expected_mse(other, phases, result["calibration"], c))))
        return {"relative_intensity_rms_change": float(relative), "expected_phase_rmse_rad": float(rmse)}
    record = {"result_sha256": file_hash(Path(run)/"result.json"), "split": "validation",
              "specimens": len(phases), "base": compare(base), "double_padding": compare(more_pad),
              "half_pitch_same_fov": compare(fine),
              "interpretation": "Diagnostic, not an automatic convergence certificate. Fine object is interpolated; detector is point sampled. Check larger windows and physically bandlimited phantoms before claims."}
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    json_write(destination, record)
    return record

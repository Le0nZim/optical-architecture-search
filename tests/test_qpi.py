"""Physical/statistical invariants, selection integrity, and frozen artifacts."""
from dataclasses import replace
import json
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from qpi.config import Config
from qpi.data import generate, import_directory, load_split, phantom, roi
from qpi.physics import (calibrate, expected_mse, intensity, orient, physical,
                         prescription, sample_camera, freeform_phase, grating_phase, tukey_profile)
from qpi.search import initial_parameters, repair, search


@pytest.fixture
def small():
    return Config(size=8, padding=4, n_train=3, n_validation=2, n_test=3,
                  population=2, generations=2, gradient_steps=1, noise_repeats=2)


def test_noise_risk_matches_independent_poisson_gaussian_monte_carlo(small):
    rng = np.random.default_rng(17)
    signal = rng.uniform(.1, 2., (8, 8)).astype(np.float32)
    truth = rng.uniform(0, .5, (8, 8)).astype(np.float32)
    cal = {"gain": .4, "offset": -.1, "inverted": True}
    photons = 8.
    analytic = float(expected_mse(signal, truth, cal, small, photons))
    electrons = sample_camera(np.broadcast_to(signal, (12000, 8, 8)), photons, small, rng)
    prediction = cal["gain"]*np.asarray(orient(electrons, True))/(photons*small.quantum_efficiency)+cal["offset"]
    empirical = np.mean((prediction-truth)[:, roi(8)]**2)
    assert empirical == pytest.approx(analytic, rel=.015)


def test_calibration_uses_physical_scale_and_penalizes_photon_loss(small):
    target = jnp.asarray(phantom(8, 19, .5))[None]
    good = 1.+target*2
    dim = good*.01
    cal_good = calibrate(good, target, small)
    cal_dim = calibrate(dim, target, small)
    good_risk = float(expected_mse(good, target, cal_good, small).mean())
    dim_risk = float(expected_mse(dim, target, cal_dim, small).mean())
    assert dim_risk > good_risk*2
    noiseless = calibrate(good, target, small, noise_aware=False)
    assert float(expected_mse(good, target, noiseless, small, noise_aware=False).mean()) < 1e-10


def test_global_phase_is_unobservable_and_passive_system_cannot_gain_power(small):
    p = initial_parameters(small, np.random.default_rng(1))
    phase = jnp.asarray(phantom(8, 2, .5))
    arch = jnp.array([1, 12, 1])
    forward = jax.jit(lambda x: intensity(p, arch, x, small))
    a, b = forward(phase), forward(phase+.73)
    np.testing.assert_allclose(a, b, rtol=2e-4, atol=3e-6)
    assert float(a.mean()) <= small.element_transmission**3 + 2e-5
    assert float(a.min()) >= 0


def test_hardware_constraints_and_export_remain_physical_under_grid_refinement(small):
    rng = np.random.default_rng(3)
    p = initial_parameters(small, rng)
    physical_p = physical(p, small)
    assert float(physical_p["gaps_um"].sum()) == pytest.approx(small.length_um, rel=1e-6)
    assert np.all(physical_p["gaps_um"] >= small.min_gap_um)
    c = replace(small, max_components=1)
    assert np.count_nonzero(repair([1, 12, 1], c, rng) != 2) == 1
    arch = np.array([9, 10, 11])
    a = prescription(p, arch, small)
    b = prescription(p, arch, replace(small, size=16, pitch_um=small.pitch_um/2, padding=8))
    assert a == b  # A convergence check must not change grating/axicon parameters.


def test_freeform_mask_has_same_physical_profile_with_more_padding(small):
    # A linear control grid has an independently known continuous profile.
    axis = np.linspace(-.5, .5, small.mask_grid)
    mask = jnp.asarray(axis[:, None] + axis[None, :], dtype=jnp.float32)
    a = np.asarray(freeform_phase(mask, small))
    b = np.asarray(freeform_phase(mask, replace(small, padding=8)))[4:-4, 4:-4]
    fine = np.asarray(freeform_phase(mask, replace(small, size=16, padding=8,
                                                   pitch_um=small.pitch_um/2)))[::2, ::2]
    coordinate = (np.arange(16)-8)*small.pitch_um/small.clear_aperture_um
    expected = np.pi*np.tanh(coordinate[:, None]+coordinate[None, :])
    np.testing.assert_allclose(a, expected, atol=1e-6)
    np.testing.assert_allclose(a, b, atol=1e-6)
    np.testing.assert_allclose(a, fine, atol=1e-6)


def test_gratings_and_tukey_profile_do_not_move_with_fft_boundaries(small):
    wide = replace(small, padding=8)
    fine = replace(small, size=16, padding=8, pitch_um=small.pitch_um/2)
    for profile in (lambda c: grating_phase(10, 93., c),
                    lambda c: grating_phase(11, 93., c),
                    lambda c: tukey_profile(21., c)):
        a = np.asarray(profile(small))
        np.testing.assert_allclose(a, np.asarray(profile(wide))[4:-4,4:-4], atol=1e-6)
        np.testing.assert_allclose(a, np.asarray(profile(fine))[::2,::2], atol=1e-6)


def test_datasets_are_reproducible_and_tampering_is_rejected(tmp_path, small):
    a = generate(small, tmp_path/"a")
    b = generate(small, tmp_path/"b")
    assert a == b
    train, ids, _ = load_split(tmp_path/"a", "train", small)
    _, val_ids, _ = load_split(tmp_path/"a", "validation", small)
    assert set(ids).isdisjoint(val_ids)
    train[0, 4, 4] += .1
    np.savez_compressed(tmp_path/"a"/"train.npz", phase=train, ids=np.array(ids))
    with pytest.raises(ValueError, match="checksum"):
        load_split(tmp_path/"a", "train", small)


def test_imported_acquisition_groups_cannot_cross_splits(tmp_path, small):
    generate(small, tmp_path/"source")
    for split in ("train", "validation", "test", "ood"):
        path = tmp_path/"source"/f"{split}.npz"
        with np.load(path) as f:
            phase, ids = f["phase"], f["ids"]
        np.savez_compressed(path, phase=phase, ids=ids, group_ids=np.array([split]*len(ids)))
    import_directory(small, tmp_path/"source", tmp_path/"valid")
    path = tmp_path/"source"/"validation.npz"
    with np.load(path) as f:
        phase, ids = f["phase"], f["ids"]
    np.savez_compressed(path, phase=phase, ids=ids, group_ids=np.array(["train"]*len(ids)))
    with pytest.raises(ValueError, match="acquisition group"):
        import_directory(small, tmp_path/"source", tmp_path/"invalid")


def test_search_never_opens_test_and_saved_winner_reproduces_score(tmp_path, small, monkeypatch):
    from qpi.__main__ import run
    from qpi.artifacts import load_result
    import qpi.__main__ as cli
    generate(small, tmp_path/"data")
    seen = []
    def guard(root, split, config):
        assert split in ("train", "validation")
        seen.append(split)
        return load_split(root, split, config)
    monkeypatch.setattr(cli, "load_split", guard)
    run(small, tmp_path/"data", tmp_path/"run", "search", 7)
    c, result, p = load_result(tmp_path/"run")
    validation, _, _ = load_split(tmp_path/"data", "validation", small)
    arch = jnp.array(result["architecture"])
    s = jax.jit(jax.vmap(lambda x: intensity(p, arch, x, c)))(validation)
    mse = float(expected_mse(s, validation, result["calibration"], c).mean())
    assert mse == pytest.approx(result["validation_mse_rad2"], rel=3e-5)
    assert seen == ["train", "validation"]
    assert result["candidate_evaluations"] == c.population*c.generations
    assert result["gradient_steps"] == c.population*c.generations*c.gradient_steps
    assert result["rejected_gradient_steps"] == 0
    assert np.all(np.diff([x["best_validation_objective"] for x in result["history"]]) <= 0)


def test_freeze_requires_complete_suite_and_detects_changed_selection(tmp_path, small, monkeypatch):
    from qpi.__main__ import freeze, evaluate_suite
    from qpi.artifacts import file_hash, json_write
    import qpi.__main__ as cli
    generate(small, tmp_path/"dataset")
    plan = {"config": small.to_dict(), "methods": ["search"], "seeds": [0],
            "dataset": str(tmp_path/"dataset"),
            "dataset_manifest_sha256": file_hash(tmp_path/"dataset"/"manifest.json")}
    json_write(tmp_path/"suite.json", plan)
    with pytest.raises(FileNotFoundError):
        freeze(tmp_path)
    path = tmp_path/"runs"/"search-seed-0"
    path.mkdir(parents=True)
    json_write(path/"result.json", {"placeholder": "artifact integrity test"})
    monkeypatch.setattr(cli, "check_run", lambda *a, **kw: None)
    freeze(tmp_path)
    json_write(path/"result.json", {"changed": True})
    with pytest.raises(ValueError, match="changed after freeze"):
        evaluate_suite(tmp_path)


@pytest.mark.parametrize("kwargs", [{"quantum_efficiency": 0}, {"max_components": 4},
    {"photons": float("nan")}, {"evaluation_photons": []}, {"read_noise_e": -1},
    {"evaluation_defocus_um": [-300.]}, {"length_um": 100.}, {"size": 7},
    {"n_test": 0}, {"train_defocus_um": [0., 0.]}])
def test_invalid_physics_fails_early(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)

"""Strict, JSON-serializable experiment settings. Lengths are micrometres."""
from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path


METHODS = ("search", "random", "restricted", "fixed", "freeform", "no_noise", "defocus", "constant")


@dataclass(frozen=True)
class Config:
    size: int = 32
    pitch_um: float = 5.0
    wavelength_um: float = 0.532
    padding: int = 64
    stations: int = 3
    length_um: float = 16000.0
    min_gap_um: float = 250.0
    focal_min_um: float = 3000.0
    focal_max_um: float = 20000.0
    width_min_um: float = 10.0
    width_max_um: float = 200.0
    min_grating_period_um: float = 40.0
    clear_aperture_um: float = 280.0
    element_transmission: float = 0.98
    max_components: int = 3
    component_penalty: float = 0.0
    mask_grid: int = 8
    phase_max_rad: float = 0.5
    data_seed: int = 20260925
    n_train: int = 32
    n_validation: int = 16
    n_test: int = 32
    population: int = 8
    generations: int = 8
    gradient_steps: int = 10
    learning_rate: float = 0.03
    mutation_probability: float = 0.5
    photons: float = 100.0
    quantum_efficiency: float = 0.8
    read_noise_e: float = 1.5
    train_defocus_um: tuple[float, ...] = (0.0,)
    evaluation_photons: tuple[float, ...] = (10., 30., 100., 300., 1000.)
    evaluation_defocus_um: tuple[float, ...] = (0., -100., 100.)
    noise_repeats: int = 4
    calibration_ridge: float = 1e-8

    def __post_init__(self):
        integers = ("size", "stations", "mask_grid", "n_train", "n_validation",
                    "n_test", "population", "generations", "gradient_steps", "noise_repeats")
        for name in integers:
            x = getattr(self, name)
            if type(x) is not int or x < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.size < 8 or self.size % 2:
            raise ValueError("size must be even and at least 8")
        if type(self.padding) is not int or self.padding < 0:
            raise ValueError("padding must be a nonnegative integer")
        if type(self.data_seed) is not int or not 0 <= self.data_seed < 2**32:
            raise ValueError("data_seed must be a uint32 integer")
        if type(self.max_components) is not int or not 1 <= self.max_components <= self.stations:
            raise ValueError("max_components must be between 1 and stations")
        for f in fields(self):
            x = getattr(self, f.name)
            if isinstance(x, float) and not math.isfinite(x):
                raise ValueError(f"{f.name} must be finite")
        positive = ("pitch_um", "wavelength_um", "length_um", "min_gap_um",
                    "focal_min_um", "focal_max_um", "width_min_um", "width_max_um",
                    "clear_aperture_um", "min_grating_period_um", "learning_rate", "photons", "phase_max_rad",
                    "calibration_ridge")
        if any(getattr(self, n) <= 0 for n in positive):
            raise ValueError("Physical scales, learning rate and ridge must be positive")
        if self.length_um <= (self.stations + 1) * self.min_gap_um:
            raise ValueError("length_um must exceed the sum of minimum gaps")
        if self.focal_min_um >= self.focal_max_um or self.width_min_um >= self.width_max_um:
            raise ValueError("Parameter bounds must be ordered")
        if not (0 < self.quantum_efficiency <= 1 and 0 < self.element_transmission <= 1):
            raise ValueError("QE and intensity transmission must be in (0, 1]")
        if self.read_noise_e < 0 or self.component_penalty < 0:
            raise ValueError("Noise and component penalty must be nonnegative")
        if not 0 <= self.mutation_probability <= 1 or self.phase_max_rad >= math.pi:
            raise ValueError("Invalid mutation probability or weak-phase range")
        if self.mask_grid > self.size + 2 * self.padding:
            raise ValueError("mask_grid exceeds working grid")
        for name in ("train_defocus_um", "evaluation_photons", "evaluation_defocus_um"):
            values = tuple(getattr(self, name))
            object.__setattr__(self, name, values)
            if not values or not all(math.isfinite(x) for x in values) or len(set(values)) != len(values):
                raise ValueError(f"{name} must contain distinct finite values")
        if min(self.evaluation_photons) <= 0:
            raise ValueError("Evaluation photon budgets must be positive")
        if self.photons not in self.evaluation_photons or 0. not in self.evaluation_defocus_um:
            raise ValueError("Evaluation must include the training dose and zero defocus (primary endpoint)")
        if min(*self.train_defocus_um, *self.evaluation_defocus_um) <= -self.min_gap_um:
            raise ValueError("Defocus must leave every final gap positive")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path):
        raw = json.loads(Path(path).read_text())
        unknown = set(raw) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        return cls(**raw)

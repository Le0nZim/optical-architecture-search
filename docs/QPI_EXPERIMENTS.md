# Noise-aware optical architecture search for relative phase imaging

## What is ready

This is a reproducible **simulation benchmark**, building on the corrected
Chromatix/JAX implementation in this repository. The initial scientific question
is whether discrete selection of conventional optical component families, with
continuous parameter optimization, improves phase accuracy under a fixed incident
photon budget and practical geometric bounds. It does not establish novelty,
clinical usefulness, experimental superiority, or a high-NA microscopy result.

The executable workflow includes synthetic datasets, custom calibrated phase-map
import, eight methods/controls, train-only readout calibration, validation-only
selection, held-out noise/dose/defocus evaluation, physical prescriptions,
checksummed artifacts, convergence diagnostics, tables, and figures.

## Installation and first experiments

Use Python 3.12 and the commands in the root README. `requirements-qpi.lock`
records the tested CPU environment; `requirements.txt` remains the more permissive
legacy installation. For a compatible NVIDIA machine, install the appropriate
GPU extra for the same JAX version after the lock file, for example:

```bash
python -m pip install 'jax[cuda13]==0.11.2'
python -c 'import jax; print(jax.devices())'
```

Consult the [official JAX installation instructions](https://docs.jax.dev/en/latest/installation.html)
for driver/platform requirements and the CUDA 12 alternative. GPU execution has
not been validated by this repository preparation. Each job uses one device;
the suite runs jobs sequentially and does not implement distributed training.

```bash
# A three-seed pilot; generates data locally with no downloads.
python -m qpi suite --config configs/qpi/pilot.json --seeds 0 1 2 --out runs/pilot

# Inspect result.json, validation curves, prescriptions and numerical convergence.
python -m qpi check-grid --run runs/pilot/runs/search-seed-0 \
  --data runs/pilot/dataset --out runs/pilot/search-seed-0-grid.json

# Freeze ONLY after the intended methods/settings are fixed.
python -m qpi freeze --suite runs/pilot
python -m qpi evaluate --suite runs/pilot
python -m qpi report --metrics runs/pilot/evaluation/metrics.csv --out runs/pilot/report
```

Repeat the grid check for every finalist, including freeform and fixed baselines.
It retains physical parameters and calibration, doubles padding, and halves
pitch at constant field of view. It compares co-located point samples, not
integrated detector pixels. Fine-grid phase maps are interpolated. A small change
in one refinement is evidence to examine, not proof of continuous-space accuracy.
Convergence tolerances must be fixed relative to the claimed effect size before
interpreting a difference between methods. Refine further if the result changes.

`paper.json` is a larger **starting configuration**, not a validated publication
protocol. After tuning on validation and checking memory/time on your GPU:

```bash
python -m qpi suite --config configs/qpi/paper.json \
  --seeds 0 1 2 3 4 --out runs/paper-study
```

The sweep has 16 candidates × 40 generations × 20 gradient steps for each of the
six optimized methods and each seed. A gradient step uses the full training set.
The constant and defocus controls have no gradient optimization. Start the pilot
first; no runtime estimate for your GPU is implied.

Re-running the identical `suite` command resumes **completed jobs** after verifying
configuration, code, dependency versions, and dataset hashes. An interrupted job
is retained under `interrupted/` and restarted from its seed; optimizer state is
not resumed mid-job. Existing results are not overwritten. Once frozen, further
search needs a new output directory. `evaluate` also resumes completed jobs.
Copies of a study moved to another machine need the absolute dataset path in
`suite.json` resolved before freezing; do not edit a frozen protocol in place.

## Optical search space and hardware limits

Every candidate has `stations` axial locations. A categorical choice at each
station selects an element or an empty station. There is free-space propagation
before, between, and after stations. Gaps are positive, have a minimum separation,
and sum to `length_um` exactly. This is a bounded linear sequence, not arbitrary
branched graph discovery or an unconstrained change in the number of stations.

The full library retains lenses, hard/soft pupils, axicons and gratings from
`oas_core.py`, and adds a circular phase disk. Propagation is now an explicit gap
rather than a categorical slot. Identity removes a component without removing
its gap. The component cap is enforced during initialization and mutation. An
optional penalty is added to validation/training MSE per installed element.

Focal lengths, widths, phase heights and gaps are transformed to bounded physical
values. Only positive focal lengths are currently available. Phase disks and
hard pupils retain hard sampled boundaries; their widths change through mutation,
not fictitious nonzero edge gradients. Other smooth parameters use Adam. Grating
period floors and axicon limits stay fixed in physical units during numerical
refinement. Freeform masks use a fixed physical aperture with bilinearly
interpolated control points and bounded phase in (-pi, pi).

All installed components have the configured clear aperture and intensity
transmission in addition to their intrinsic mask attenuation. Empty stations
have no mount aperture or component loss. The `constant` control uses no signal.
These are **bounded ideal components**, not parts selected from a vendor catalog.
The same transmission value for all families is a placeholder to replace with
measured or catalog-specific values before a hardware claim. `width_um` follows
the component's native definition: diameter for circular pupils/phase disks,
side length for a square, Gaussian waist for a Gaussian pupil, and the existing
Chromatix scale for super-Gaussian pupils. The QPI Tukey pupil has a flat core of
diameter `width_um` and a cosine taper to the fixed clear aperture. Its support
and the grating origins are fixed in physical coordinates rather than relative
to the FFT window, so changing numerical padding cannot change these optics.

## Phase, photons, camera, and readout

Objects have unit amplitude and known relative phase, with a zero-phase outer
background. Default phantoms are smooth positive profiles with peak phase up to
0.5 rad; the OOD set uses sharp ellipses at the same phase range. OOD here tests
shape, not a new tissue type or a new amplitude distribution. A fixed central
square ROI is used for every image and method; it is not estimated from truth.
Global phase is not observable from intensity. These experiments do not recover
arbitrary absolute or 0–2pi unwrapped phase.

The illumination is a uniform finite square on the input grid. The numerical
padding contains zero field and adds no photons. Scalar, coherent, monochromatic,
paraxial propagation is padded once and cropped once. The current sensor uses
point samples; it does not integrate over a pixel's active area. The finite square
illumination and hard phase/pupil edges need particular care in convergence tests.

Let `I` be intensity divided by the **fixed incident intensity**, `D` the incident
photons per input pixel, `q` detector QE, and `s` read-noise standard deviation in
electrons. With one exposure per specimen:

```
Y ~ Poisson(D q I) + Normal(0, s²)
J = Y / (D q)
phase_hat = a J + b
E[(phase_hat - phase)²] = (a I + b - phase)² + a² [I/(D q) + s²/(D q)²]
```

This expectation is exact under this camera/affine-readout model, not a Gaussian
approximation to shot noise. Total specimen illumination is `D × size²` photons,
even if a design discards almost all light. There is no per-image variance
normalization, target normalization, or output-photon renormalization. Electrons
are simulated with exact Poisson sampling for the independent Monte Carlo check.
Negative read-noise values are retained; saturation, dark current, clipping,
quantization, detector correlations and calibration uncertainty are not modeled.

Two scalar readout coefficients and an upright/inverted coordinate convention
are fitted on training images. The same options are available to every method.
The gain is covariance divided by intensity variance plus expected noise variance
and a small numerical ridge. This fitted readout is differentiated through during
optical training. It is saved and **never fitted to validation/test truth** or
refitted at a new dose/defocus. The inversion is about the optical axis, with
boundary wrap outside the scoring ROI. This is a deliberately limited calibrated
phase-contrast estimator, not a general nonlinear phase reconstruction network.

## Baselines and budget accounting

| Method | Purpose | Search/training |
|---|---|---|
| `search` | Full component-library hypothesis | Discrete mutation, continuous Adam, inherited weights |
| `random` | Value of evolutionary selection | Fresh random architectures/parameters each generation, same local Adam budget |
| `restricted` | Value of the wider library | Lens, identity, circular pupil, phase disk; otherwise same search |
| `fixed` | Value of categorical topology changes | Lens–phase-disk–lens fixed sequence; all allowed continuous parameters and hard widths optimized; first candidate initialized near a 4f layout |
| `freeform` | Competing flexible phase optics | One learned phase mask at every station, optimized gaps, same affine readout |
| `no_noise` | Whole noise-aware objective ablation | Shot/read variance omitted from training, calibration and selection; identical noisy evaluation |
| `defocus` | Simple optical control | No installed components, fixed total propagation, trained affine readout |
| `constant` | Dataset-prior control | Training ROI mean phase, zero intensity gain |

The first six use the same population, generations, gradient steps, training
images, validation images, exposure count, total length, illumination and camera.
The fixed/freeform baselines require the full station budget; complexity sweeps
with fewer allowed components should explicitly select the applicable methods.
The freeform control has many more parameters and different fabrication needs.
The fixed baseline is a matched ideal phase-contrast layout, not a complete
model of a commercial Zernike microscope. The freeform baseline is not a
reproduction of Mengu & Ozcan's published network. Published-method reproduction
remains a separate experiment.

Adam moments reset at each generation for every candidate, including elites;
weights are inherited. Children receive bounded parameter perturbations and may
change one categorical operation. An unchanged best design is archived before
breeding. Readout calibration is analytic and has no iterative fit budget.
`result.json` records candidate evaluations, gradient steps, rejected updates,
timing including compilation, and configuration. Equal gradient budgets do not
imply equal wall time, parameter counts, or total hyperparameter-tuning effort.
Report all three when making a comparison. More starts or specialized optimizers
may be needed to establish that each baseline has been competently optimized.

## Robustness, dose, and complexity studies

`robust.json` trains over nominal and +/-100 um sensor defocus. Its comparison
against the nominal pilot evaluates the benefit of that specific robustness
training, with its extra simulation cost disclosed. A design's readout stays
fixed when the sensor moves. This first tolerance model does not cover component
decentering, tilt, fabrication errors, aberrations, or wavelength changes.

The standard dose plot evaluates one trained design at several doses; it is not
a separately optimized design at each dose. For dose-specific design, copy a
config and change `photons` (keeping it in `evaluation_photons`), then create a
new suite with the same dataset. Sweep `max_components` or `component_penalty`
in separate suites for complexity studies. Do not combine differently specified
studies into one result table without identifying the differing constraints.

## Importing phase truth and protecting the split

Prepare `train.npz`, `validation.npz`, `test.npz`, and `ood.npz`. Each contains:

| Key | Meaning |
|---|---|
| `phase` | Finite floating array `(N, size, size)`, in radians, magnitude below pi, zero known background outside the ROI |
| `ids` | Unique string specimen/patch identifiers |
| `group_ids` | String acquisition/patient/specimen-group identifiers; repeats within a split allowed, never across splits |

```bash
python -m qpi import-data --config configs/qpi/pilot.json \
  --source /path/to/presplit-phase-maps --out runs/measured-phase-data
python -m qpi suite --config configs/qpi/pilot.json --data runs/measured-phase-data \
  --seeds 0 1 2 --out runs/measured-phase-study
```

Calibrate phase/background and physical pitch independently before import. Do not
normalize each image to unit range. Partition acquisition groups before extracting
patches, and explain how the OOD cohort differs. Import checks specimen identifiers,
exact specimen hashes, acquisition groups and dataset integrity. It cannot detect
undisclosed related specimens with different IDs/groups. Imported measured phase
maps still produce **simulated camera measurements**, not a hardware experiment.

Search opens only train and validation arrays. The manifest stores split metadata
and checksums; the test arrays are opened only by evaluation. `freeze` checks that
every planned job is complete and seals the selected result hashes before the
evaluation command will run. This is a reproducibility convention, not security
against intentional test peeking or an external preregistration service. If test
results influence redesign, report that exploration and acquire another holdout.

## Results and paper workflow

`result.json` contains the prescription, calibration, validation history, code
hashes (including vendored optics), commit, package versions and device. Raw
parameters are in `parameters.npz`. `evaluation/metrics.csv` has one row per
specimen/seed/dose/defocus, with exact and sampled phase MSE, bias, throughput and
detected electrons. Example phase/prediction arrays are saved at nominal settings.

The report exports `summary.csv`, paired differences, `results_table.tex`, and
`dose_accuracy.pdf/png`. RMSE is computed per search seed and averaged. Descriptive
95% bootstrap intervals resample search seeds conditional on the fixed test set;
one-seed intervals are left blank. These do not represent biological-population
uncertainty. Paired comparisons match seeds; noise repeats/pixels are not treated
as independent specimens. Five seeds are a starting point, not a power analysis.

See [paper/README.md](../paper/README.md) for manuscript commands and
[RESEARCH_PLAN.md](RESEARCH_PLAN.md) for the evidence still needed before a strong
publication claim.

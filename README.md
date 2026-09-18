# Optical Architecture Search (OAS)

A research prototype for choosing a sequence of scalar optical elements and
optimizing their parameters with Chromatix, JAX and Optax. The primary search is
memetic: evolutionary selection of discrete elements plus gradient updates for
continuous, differentiable parameters. It is a heuristic, not a guarantee of an
optimal or uniquely identifiable optical system.

## Install and run

Python 3.12 or newer is required. Run these commands from the repository root:

```bash
python3.12 -m venv venv312
source venv312/bin/activate
python -m pip install -r requirements-gui.txt
streamlit run gui.py
```

For headless tests and an offline synthetic demonstration, without PyTorch,
CIFAR-10 downloads, or Streamlit:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python run_search.py --task amplitude --output amplitude_result.json
python run_search.py --task phase --output phase_result.json
```

The GUI test is skipped when Streamlit is absent. CI installs Streamlit and
runs it with local synthetic images. CIFAR-10 requires torchvision and network
access on its first use. GPU installation of JAX is platform-specific; the
commands above support the CPU path used for verification.

## Search and evaluation

1. A seed controls initialization, image selection, parameter perturbations and
   architecture mutations.
2. Each candidate takes gradient steps on training images. Distances/focal lengths
   and widths use separate learning rates and are projected to positive bounds.
3. **Updated** candidates are scored on the same fixed validation images at every
   generation. Non-finite scores cannot win, and non-finite updates are rejected.
4. A copy of the best evaluated architecture and its parameters is retained before
   breeding. Unchanged elites retain their optimizer state; perturbed children get
   fresh Adam state. The returned result is the saved winner.
5. The GUI evaluates that winner on 16 images from the **official CIFAR-10 test
   split**, reports loss and sensor throughput, and exports the design/settings as
   JSON. Training and validation use disjoint partitions of the training split
   (16 training / 8 validation images in this small demonstration).

`oas_core.py` contains the testable simulation, objectives and search. `gui.py`
only supplies the interface. `oas_data.py` provides deterministic dataset splits.
`run_search.py` is a small synthetic execution check, not a performance benchmark.

## Optical model and tasks

All lengths are in **micrometres**. The model uses coherent, monochromatic,
paraxial scalar optics. The object/sensor grid defaults to 64 × 64 at 5 µm pitch,
with 0.532 µm wavelength and 64 zero-padding pixels per side. The field is padded
once and cropped at the sensor; free-space segments do not insert hidden stops.

**Amplitude imaging:** the ideal infinite-aperture 4f system has magnification
`M = f2/f1`. Its amplitude on the candidate's fixed sensor coordinates is
`u_out(r) = u_in(-r/M)/M`, up to an irrelevant global phase. The target evaluates
this mapping by linear interpolation with zero outside the object support,
then takes intensity. This preserves magnification, inversion, and the `1/M²`
intensity factor. Comparing two `ff_lens` arrays directly with fixed-grid
propagation was incorrect when their physical pixel spacings differed.
The objective is relative intensity MSE with a small denominator floor.

**Phase imaging:** the object introduces `0.5 * (object - mean(object))` radians.
The objective compares standardized output intensity with standardized relative
object phase. A variance floor inside the square root keeps flat-image gradients
finite. This is a **weak relative phase-contrast** task. An intensity-only linear
system cannot observe a global phase offset; arbitrary 0–2π unwrapped phase cannot
be uniquely inferred. The contrast objective is scale-invariant and does not
optimize photon efficiency or noise robustness. Throughput is reported separately.

## Element library

| Operation | Parameter / treatment |
| --- | --- |
| `Propagate` | Positive distance `z`, gradient optimized |
| `ThinLens` | Positive focal length `f`; converging lenses only |
| `Identity` | No parameter |
| `CircularPupil`, `SquarePupil`, `RectangularPupil` | Exact sampled hard masks; width changes through **evolution**, since Boolean edges have zero width gradient almost everywhere |
| `GaussianPupil`, `SuperGaussianPupil`, `TukeyPupil` | Gradient-optimized width; Tukey full-width limit is finite |
| `Axicon` | Bounded slope angle derived from `w`, exported in radians |
| `SawtoothGrating` | Period at least 8 pixels; one-wave phase depth |
| `SinusoidGrating` | Period at least 8 pixels; π-radian sinusoidal phase amplitude |

Grating phase uses physical period and thickness, not the maximum phase sampled
on the current grid. The lens lower bound limits the lens's own phase increments
on the padded grid. It does not guarantee that a composed optical system is
adequately sampled. JSON export reports effective parameters and their units.

## Interpreting a result

A low intensity error does **not** establish that the search rediscovered a unique
4f layout. Ideal amplitude targets depend on `f2/f1`, not the two focal lengths
independently; consecutive propagation segments and consecutive lenses also
have equivalent compositions. There is no fixed object-to-sensor distance,
component-count penalty, lens catalog, aberration model, or fabrication tolerance
in this prototype.

Before using a design as a research result, assess multiple seeds and a larger
held-out dataset, sensor noise and throughput, and convergence under finer pixel
sampling and larger computational padding. FFT propagation still has periodic
outer boundaries. Local sampling checks are warnings, not a proof of convergence.
The target is ideal and may not be realizable with a particular finite grid or
search space. Hard apertures and phase discontinuities need particular care.

## Other scripts

- `discrete_oas.py`: experimental straight-through Gumbel search. Its forward pass
  selects one operation, but its architecture gradient remains a biased relaxed
  estimator. Final evaluation is deterministic argmax, matching its printed design.
- `inverse_design.py`, `oas_inverse_design.py`: fixed-topology, freeform phase-mask
  optimization. They share the corrected sensor-grid target and positive distance
  constraints. An intensity match does not imply their masks are thin lenses.
- `task5_1.py`: ideal Fourier-optics demonstration on its **native output sampling**.
- `debug_target.py`: comparison with the same sensor-grid target; its deliberately
  small legacy focal lengths produce a sampling warning.

Soft coherent sums of candidate paths describe a different forward model from a
single selected path and can create a discretization gap. That motivates the
memetic implementation; it is not evidence that every differentiable architecture
search method must fail or that gradient information is universally destroyed.

See [AUDIT.md](AUDIT.md) for the defects, fixes, validation evidence and remaining
limitations. Existing PNG examples predate this audit and are not regenerated
benchmarks of the corrected implementation.

## License

MIT. The repository includes a vendored Chromatix distribution with local patches;
its license is retained in `chromatix/LICENSE`.

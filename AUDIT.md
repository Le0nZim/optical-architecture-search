# Optical architecture search audit

Audited the upstream and fork at `d4c132696dc25134b2f17bb5dd9996f0c360eef1`.
Both repositories had the same source at the start. Changes target the fork's
`main` branch. This review covers the application, its optical/optimization
assumptions, the vendored functions it exercises, and the existing library tests;
it is not a proof that every vendored scientific routine is correct.

## Confirmed defects and corrections

| Severity | Finding in the original code | Correction |
| --- | --- | --- |
| Blocker | A fresh editable installation could not import Chromatix: `utils/__init__.py` imported the absent `chromatix.data.data`. This also prevented test collection. | Remove the unused sample-data re-export. Anchor the root `/data/` ignore rule so it does not hide future vendored source directories. Add root installation instructions and CI. |
| High | The GUI sorted the old population, bred a new population, then used an index from the old population to display/evaluate a member of the new one. It could return a different, unscored architecture. | Save a copy of the evaluated best architecture and its exact parameters before reproduction; retain the best across generations; do not breed after the final evaluation. |
| High | Fitness was the loss returned **before** the final optimizer update, on a single image that changed each generation. Scores were not comparable across generations and did not describe the displayed parameters. | Recompute fitness **after** updates on a fixed validation batch. Test that evaluating the saved winner reproduces its reported validation score. |
| High | The two ideal `ff_lens` calls changed the target's pixel pitch to `dx*f2/f1`, but candidates retained `dx`. The code compared arrays at different physical coordinates. At `dx=5`, `f1=2000`, `f2=4000`, the target pitch is 10 µm while the candidate pitch is 5 µm. | Use the analytic coherent 4f amplitude map `u(-r/M)/M` on the candidate sensor coordinates; interpolate amplitude before squaring and zero outside object support. This also removes periodic target wrap at the image border. |
| High | The GUI amplitude denominator added `1e-8` to physical intensity-squared values around `1e-12` to `1e-10`. This dominated the intended relative normalization. For a uniform 0.5 amplitude object at the default grid, a completely dark prediction scored about 0.000596 instead of relative MSE 1. | Use relative MSE with a `1e-20` floor and enforce equal image shapes. Cover low-intensity and zero-target cases. |
| High | Each propagation padded and immediately cropped to the object grid. A sequence of free-space segments therefore inserted unintended field stops, discarding light that could return later. | Pad the working field once and crop only at the sensor. Verify `P(z1)P(z2)` against `P(z1+z2)`. Freeform phase plates occupy their original footprint and apply zero additional phase outside it. |
| High | `std(image) + epsilon` has an undefined derivative at exactly constant output because the square-root derivative is evaluated before epsilon is added. Such outputs occur for identity/phase-only layouts. | Put the variance floor inside the square root and normalize intensity units before the phase contrast calculation. Verify finite batched gradients across every library operation. |
| High | Arbitrary object values mapped to 0–2π, while the requested intensity target represented unwrapped phase. Values 0 and 1 have identical complex transmission, and global phase is unobservable under linear propagation plus intensity detection. | Explicitly scope the demonstration to weak relative phase contrast, using a 0.5-rad range and a centered, standardized target. Document that this is not absolute or arbitrary unwrapped phase recovery. |
| Medium | The purported test image was sampled with `train=True`; training/test samples could overlap. NumPy population/mutation and PyTorch shuffling were unseeded despite the JAX seed. | Use deterministic, disjoint train/validation partitions and the official CIFAR-10 test split. A single seed controls the main search. Test complete-run reproducibility. |
| Medium | Gumbel evaluation at temperature 0.01 was still random. Dividing logits plus Gumbel noise by any positive temperature does not change their argmax. The output could disagree with the printed architecture. | Evaluate the exported argmax architecture without Gumbel noise. Split training-image and routing random keys independently. |
| Medium | Grating phase was divided by its sampled maximum. The effective optic therefore depended on the grid; sampled sawtooth zeros produced `0/0`. The original thickness/period coupling also generated steep, aliased phase masks. | Normalize sawtooth position by the physical period and use the sinusoid's analytic unit amplitude. Use a minimum 8-pixel period and bounded phase depths in the search. Bound the axicon slope. |
| Medium | The Tukey formula divided by `1-alpha`, singular when pupil width reached field extent. An inactive singular branch is also dangerous when `vmap` batches `lax.switch`. | Clip the width fraction and keep the taper denominator finite at its full-width limit. Test values and derivatives at and beyond full width. |
| Medium | Forward propagation used `abs(raw)+0.001`, while reports printed raw values. Widths also received the same 50-µm Adam step size as distances. Children inherited Adam moments even after architecture and parameter changes. | Project and store positive physical parameters; report those values and correct units. Use separate width/distance learning rates; initialize fresh optimizer state for perturbed children. Reject non-finite updates and fitness. |
| Medium | UI counts and target focal lengths could be zero/negative, producing empty/uninitialized search state or singular optics. | Validate configuration before compilation and enforce widget bounds. Population size one and one generation are supported and tested. |
| Documentation | Hard pupil widths were advertised as differentiable, and descriptions implied the optimizer would recover a unique classical layout. | State that exact sampled hard-edge widths are evolved, not learned by their zero Boolean-mask gradient; distinguish heuristic inverse design from identifiability or optimality claims. |

The suspected phase-array broadcasting mismatch was checked and **not found**:
this vendored scalar `Field.intensity` is already a two-dimensional array. The
corrected objectives nevertheless reject unequal shapes rather than silently
broadcasting them.

## Validation performed

Fresh Python 3.12 environment, CPU backend. Installed the repository's vendored
Chromatix and real JAX/Optax dependencies; no physics mocks were used in the
numerical or search tests.

- `python -m pytest -q --disable-warnings`: **95 passed, 2 skipped** in about 80 s.
- The two skips were already present in the vendored suite: a SciPy-dependent
  Gaussian-filter comparison and a thick-sample absorption test marked as having
  unresolved mathematics. No new regression test was skipped in this environment.
- Regression coverage includes analytic off-axis Gaussian targets at 0.5×, 1×,
  and 2× magnification; target border behavior; free-space composition; all 12
  operators under batched differentiation; flat contrast; physical grating
  formulas; deterministic export; saved winner/score consistency; dataset split
  selection; invalid inputs; and repeated seeded searches.
- A Streamlit `AppTest` completes a real search and evaluation with local images,
  renders the results, and retains them on rerun. Only the CIFAR loader is replaced
  for this UI test, avoiding an external dataset download.
- Existing continuous inverse-design examples have forward/gradient checks using
  their shared corrected target. The experimental Gumbel path has a deterministic
  export check.
- `git diff --check` and Python compilation checks passed.

Two small, offline end-to-end searches were also run with seed 42, a 16×16 grid,
three slots, four candidates, two gradient steps per generation and two generations:

| Task | Best validation loss | Held-out synthetic test loss | Sensor throughput |
| --- | ---: | ---: | ---: |
| Amplitude | 0.920721 | 0.752959 | 82.79% |
| Weak relative phase | 0.999524 | 1.019983 | 100.00% |

Reproduce them with `python run_search.py --generations 2` and
`python run_search.py --task phase --generations 2`. These runs demonstrate that
the pipeline executes and exports an evaluated design; the phase result is near
the flat-output baseline and **does not demonstrate successful phase imaging**.
No full 60-generation CIFAR search, GPU benchmark, multi-seed performance study,
or experimental optical validation was performed.

Tested package versions: JAX/JAXlib 0.11.2, Optax 0.2.8, Equinox 0.13.8,
NumPy 2.5.3, pytest 9.1.1, Streamlit 1.64.0.

## Remaining model/research limitations

1. **Identifiability:** the ideal infinite-aperture target depends only on `f2/f1`.
   It cannot identify the two focal lengths independently. Consecutive free-space
   segments combine by adding distances; adjacent thin lenses combine their
   optical powers. Identity slots introduce more equivalent descriptions.
   Image agreement does not prove recovery of a unique physical layout.
2. **Sampling and boundaries:** FFT propagation remains periodic at the outer
   computational boundary. The lens bound and local sampling warnings address
   particular undersampling risks, not all accumulated spatial bandwidth or
   wraparound. Check a fixed physical design with finer sampling and more padding.
   Hold its actual parameters fixed during that comparison; rerunning the search
   changes both the design and the conservative lens bounds. An unflagged design
   is not automatically converged.
3. **Target approximation:** amplitude interpolation is not sensor pixel
   integration or an antialiasing treatment for strong demagnification. The target
   is ideal/infinite-aperture, while candidates have finite computational support.
   A target may be unattainable with a given grid and element library.
4. **Phase and photons:** the phase objective measures relative structure and is
   largely insensitive to attenuation. Throughput is reported separately; shot
   noise, read noise, exposure, dynamic range and minimum useful photon flux are
   not optimized. Absolute phase is not recovered.
5. **Physical constraints:** the search has no required total length, lens catalog,
   fabrication tolerances, aberration/dispersion model, component-count cost, or
   vector/nonparaxial propagation. It currently allows converging lenses only and
   ties some phase-element degrees of freedom to a single scalar parameter.
6. **Generalization and optimization:** the default demonstration uses very small
   training/validation/test subsets. Validation is used adaptively for search;
   the untouched test split is essential. Longer runs, larger samples, multiple
   seeds and appropriate baselines are needed to claim a useful architecture.
   The experimental Gumbel estimator remains biased; memetic search is heuristic.

The old output PNGs were left as historical examples and are explicitly labeled
as predating these corrections in the README.

# Repository preparation validation — 2026-09-25

Verified on Linux, Python 3.12.14, CPU JAX/JAXlib 0.11.2, Optax 0.2.8 and the
vendored Chromatix 0.5.0. The exact Python dependency set is in
`requirements-qpi.lock`. These checks validate execution and specified discrete
model invariants; they are not a scientific benchmark or hardware demonstration.

- `python -m pytest tests -q`: **52 passed, 1 skipped**. The skipped module is
  the optional Streamlit GUI test because Streamlit was not installed in this
  environment. The existing GUI CI installation remains unchanged.
- QPI checks include Monte Carlo agreement with the analytic Poisson–Gaussian
  expected error, loss of information under attenuation, global-phase invariance,
  passive power bounds, physical gap/component limits, fixed physical mask and
  grating coordinates under grid changes, dataset/group integrity, no test-array
  access during search, checkpoint score reproduction, and frozen-selection
  tamper detection.
- The `smoke.json` suite completed all **eight methods**, seed 0, two generations
  for optimized methods, followed by protocol freeze, held-out evaluation and
  report generation. No gradient update was rejected. The resulting metrics had
  **192 specimen/condition rows**: 8 methods × 2 splits × 3 specimens × 2 photon
  budgets × 2 defocus settings.
- The report generated CSV summaries, matched-seed comparisons, a LaTeX table
  and PDF/PNG dose plot. The manuscript compiled with both placeholders and
  generated smoke inputs; rendered pages were inspected. Smoke-generated inputs
  are not committed as paper results.
- The fixed-design grid diagnostic ran successfully. As expected for the tiny
  8×8 execution grid, it **did not demonstrate convergence**: the searched smoke
  design's relative intensity RMS change was about 0.91 on doubling padding and
  0.48 on halving pitch. Those numbers are a warning against using this smoke
  design as a physical result, not evidence of search performance. Larger pilot
  and final designs require their own convergence studies before interpretation.

Not run here: the full multi-seed pilot/paper sweeps, GPU validation, published
baseline reproductions, or a physical experiment. The experiment guide explains
how to start these studies and which assumptions require measured calibration.

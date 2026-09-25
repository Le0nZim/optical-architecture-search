# Study plan and claim boundaries

Working direction: **noise-aware optical architecture search for quantitative
relative-phase imaging under a fixed illumination and hardware budget**.

## Primary question

At a prespecified specimen photon dose and optical envelope, does searching
component families and their sequence improve held-out relative-phase accuracy
over well-optimized fixed layouts, random architecture proposals and restricted
libraries? Can a useful advantage survive measured hardware errors?

The current code makes the simulation question executable. A narrow combination
of search space, physics and objective may be distinctive, but no claim of being
first should appear without an updated literature audit near submission. Do not
claim the first optical architecture search, first differentiable optical design,
first optical phase recovery, or first differentiable microscopy framework.

## Experiments and evidence to collect

| Stage | Runnable starting point | Evidence / decision |
|---|---|---|
| Numerical checks | `smoke.json`, tests, `check-grid` | Conservation/global-phase invariance, exact noise-risk agreement, no accidental change of physical design under numerical refinement |
| Pilot | `pilot.json`, eight methods, three seeds | Does anything beat the constant and defocus controls? Does search improve as budget increases? Are the fixed/freeform controls competitive and optimized? |
| Sampling study | Frozen finalists, larger padding and finer pitch | Numerical error must be small relative to the claimed method difference; check all methods rather than only the proposed one |
| Main simulated study | Validated revision of `paper.json`, at least five starts | Prespecified dose/ROI/phase range, per-specimen errors, search variability, throughput, budgets and failures; enough images/seeds for the target effect size |
| Noise ablation | `search` vs `no_noise` | Effect of noise-aware optical training, readout fitting and selection together; isolate these factors further if attributing an effect specifically to architecture |
| Tolerance study | `robust.json` plus nominal controls | Fixed-readout response to sensor defocus; extend to measured component placement/fabrication tolerances before claiming broader robustness |
| Complexity study | Separate component caps/penalties | Accuracy versus actual build complexity; calibrate losses/costs per component family rather than assuming equal costs |
| Data generalization | Import independent calibrated phase maps | Group-disjoint acquisitions, known phase truth, realistic phase and amplitude distributions; shape phantoms alone do not establish biological relevance |
| Published-method baselines | Separate validated reproductions | QPI diffractive design and appropriate conventional phase microscopy; current generic controls are not literature reproductions |
| Physical validation | Exported prescription plus calibrated bench | Measured photon budget, ground-truth phase standard, detector calibration, aligned optical geometry, repeat builds/acquisitions, held-out specimens |

## Physical demonstration to design with Dushan

Choose an achievable specimen-plane resolution/FOV and map the simulated input
plane to the actual microscope/relay magnification. The default 5 um grid is a
simulation-plane sampling choice, not a demonstrated microscope resolution.
Agree on available optics and axial travel before making a component catalog.
If a DMD is used, its amplitude encoding, diffraction orders, efficiency and
reference-light requirements need a calibrated model; an ideal phase disk or
continuous phase mask does not automatically describe a DMD implementation.

Measure camera gain, read variance, QE or absolute photon calibration, saturation,
dark signal and exposure timing. Measure element transmissions and alignment
uncertainties. Validate the forward model against simple propagation and known
phase targets before transferring a searched prescription. Keep reference
acquisitions and any calibration exposure costs explicit. Current single-exposure
budgets exclude that additional hardware calibration effort.

The strongest paper centerpiece would be a repeatable improvement in phase
accuracy at a specified incident dose, or an equivalent reduction in dose at a
specified accuracy, on real specimens against strong controls. A broader library
and a working optimizer alone would support a software/methods contribution of
more limited scope. No advantage or journal outcome is assumed in the manuscript.

## Writing now

Write the problem, constrained design space, measurement model, exact expected
risk, data protocol and evaluation plan now. Keep results, effect sizes,
biological interpretations, novelty superlatives and hardware claims open until
the required evidence exists. Maintain a log of pilot decisions and all test
unblindings. The paper scaffold includes citations for related work and does not
turn internal smoke outputs into a scientific result.

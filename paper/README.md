# Manuscript workspace

`main.tex` is a compilable working manuscript with implemented methods, cited
related work, and visible TODOs. It contains no invented numerical results or
claim of being first. Confirm authorship and affiliations with collaborators.

From the repository root (a LaTeX distribution with `pdflatex` and `bibtex`):

```bash
make -C paper
```

The output is `paper/build/main.pdf`. Build files are ignored by Git.
After a frozen, evaluated study, generate measured inputs directly:

```bash
python -m qpi report --metrics runs/paper-study/evaluation/metrics.csv --out paper/generated
make -C paper
```

`paper/generated/` must not already exist; choose a new report directory for
another study and preserve earlier outputs. The draft includes the table and
figure when available and otherwise displays explicit placeholders. A smoke
report only checks the writing pipeline; do not describe it as a benchmark.

The BibTeX file includes the central competing approaches and the simulation
library. Expand the related-work coverage and recheck novelty before submission.
The freeform and fixed-layout implementations are generic matched controls,
not validated reproductions of those published systems.

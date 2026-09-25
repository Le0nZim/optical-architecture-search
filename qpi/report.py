"""Generate measured CSV/LaTeX tables and figures; never insert invented results."""
from collections import defaultdict
import csv
from pathlib import Path
import numpy as np


def interval(values, rng):
    if len(values) < 2:
        return "", ""
    means = np.mean(rng.choice(values, (2000, len(values)), replace=True), axis=1)
    return tuple(float(x) for x in np.quantile(means, [.025, .975]))


def write_csv(path, rows):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def report(metrics, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with Path(metrics).open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("Metrics file is empty")
    groups = defaultdict(lambda: defaultdict(list))
    throughput = defaultdict(list)
    for row in rows:
        key = (row["method"], row["split"], float(row["photons_per_input_pixel"]), float(row["defocus_um"]))
        groups[key][int(row["seed"])].append(float(row["expected_mse_rad2"]))
        throughput[key].append(float(row["sensor_throughput"]))
    rng = np.random.default_rng(1701)
    records, per_seed = [], {}
    for key, seeds in sorted(groups.items()):
        values = {seed: float(np.sqrt(np.mean(mses))) for seed, mses in sorted(seeds.items())}
        per_seed[key] = values
        low, high = interval(list(values.values()), rng)
        records.append(dict(zip(("method", "split", "photons_per_input_pixel", "defocus_um"), key)) |
            {"mean_seed_rmse_rad": float(np.mean(list(values.values()))),
             "ci95_low_rad": low, "ci95_high_rad": high, "search_seeds": len(values),
             "specimens_per_seed": min(map(len, seeds.values())),
             "mean_sensor_throughput": float(np.mean(throughput[key]))})
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    write_csv(destination/"summary.csv", records)
    paired = []
    for (method, split, photons, dz), seeds in per_seed.items():
        search_seeds = per_seed.get(("search", split, photons, dz), {})
        common = sorted(seeds.keys() & search_seeds.keys())
        if method == "search" or not common:
            continue
        delta = [search_seeds[s]-seeds[s] for s in common]
        low, high = interval(delta, rng)
        paired.append({"baseline": method, "split": split, "photons_per_input_pixel": photons,
                       "defocus_um": dz, "mean_search_minus_baseline_rmse_rad": float(np.mean(delta)),
                       "ci95_low_rad": low, "ci95_high_rad": high, "paired_seeds": len(common)})
    if paired:
        write_csv(destination/"paired_comparisons.csv", paired)
    nominal = [r for r in records if r["split"] == "test" and r["defocus_um"] == 0]
    if nominal:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for method in sorted({r["method"] for r in nominal}):
            subset = sorted([r for r in nominal if r["method"] == method], key=lambda r:r["photons_per_input_pixel"])
            x = np.array([r["photons_per_input_pixel"] for r in subset])
            y = np.array([r["mean_seed_rmse_rad"] for r in subset])
            ax.plot(x, y, marker="o", label=method.replace("_", " "))
            if all(r["ci95_low_rad"] != "" for r in subset):
                ax.fill_between(x, [r["ci95_low_rad"] for r in subset],
                                 [r["ci95_high_rad"] for r in subset], alpha=.12)
        ax.set(xscale="log", xlabel="Incident photons / input pixel", ylabel="Phase RMSE (rad)",
               title="Simulated measurements; held-out phase maps")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(destination/"dose_accuracy.pdf")
        fig.savefig(destination/"dose_accuracy.png", dpi=180)
        plt.close(fig)
        lines = [r"\begin{tabular}{lrrr}", r"\hline", r"Method & Photons/pixel & RMSE (rad) & Seeds \\", r"\hline"]
        for r in nominal:
            method = r["method"].replace("_", r"\_")
            lines.append(f"{method} & {r['photons_per_input_pixel']:g} & {r['mean_seed_rmse_rad']:.5f} & {r['search_seeds']} " + r"\\")
        lines += [r"\hline", r"\end{tabular}"]
        (destination/"results_table.tex").write_text("\n".join(lines)+"\n")
    (destination/"README.md").write_text(
        "# Generated results\n\nRMSE is computed per search seed, then averaged across seeds. "
        "95% percentile-bootstrap intervals resample search seeds (2,000 draws), conditional on the fixed test specimens. "
        "With one seed the interval is intentionally blank; tiny pilot seed counts are not strong inferential evidence. "
        "Paired differences match search seeds; negative values favor search. These are descriptive comparisons, "
        "not corrected hypothesis tests. Noise repeats and pixels are not counted as independent specimens. "
        "Exact expected Poisson–Gaussian MSE is primary; metrics.csv also contains sampled-camera MSE.\n\n"
        "Scalar paraxial optics, point-sampled detector. Default data are synthetic phantoms; "
        "imported calibrated phase maps are also supported (see dataset_kind in metrics.csv). "
        "These figures describe simulated measurements, not experimental microscopy results.\n")
    return records

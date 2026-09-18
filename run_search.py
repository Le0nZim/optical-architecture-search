"""Small, offline end-to-end run: python run_search.py --task phase."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import jax
import numpy as np

from oas_core import AMPLITUDE, PHASE, SearchConfig, describe_architecture, evaluate, run_search, sampling_warnings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("amplitude", "phase"), default="amplitude")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("oas_smoke_result.json"))
    args = parser.parse_args()
    config = SearchConfig(shape=(16, 16), pad_width=16, num_blocks=3, pop_size=4,
                          gd_steps=2, generations=args.generations, seed=args.seed,
                          task=AMPLITUDE if args.task == "amplitude" else PHASE)
    rng = np.random.default_rng(args.seed)
    # Separate draws; interpolated smooth objects are less dominated by Nyquist frequencies.
    images = np.asarray(jax.image.resize(rng.random((12, 4, 4)).astype(np.float32),
                                        (12, 16, 16), method="linear"))
    result = run_search(config, images[:6], images[6:9],
                        lambda g, loss: print(f"Generation {g}: validation={loss:.6g}", flush=True))
    performance = evaluate(result, config, images[9:])
    record = {"config": asdict(config), "architecture_indices": result.architecture.tolist(),
              "parameters": {name: p.tolist() for name, p in zip(("z", "f", "w"), result.parameters)},
              "architecture": describe_architecture(result.architecture, result.parameters,
                                                     config.dx, config.spectrum),
              "validation_loss": result.validation_loss, "test_loss": performance["loss"],
              "test_throughput": performance["throughput"], "history": result.history,
              "sampling_warnings": sampling_warnings(result.architecture, result.parameters,
                  config.shape, config.dx, config.spectrum, config.pad_width)}
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Test loss={performance['loss']:.6g}; throughput={performance['throughput']:.4f}")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

"""Run with python -m qpi; selection and test evaluation are separate commands."""
import argparse
import csv
import json
from pathlib import Path
import time
from .artifacts import file_hash, json_write, load_result, provenance, save_result
from .config import Config, METHODS
from .data import generate, import_directory, load_split
from .search import search, validate_method


def canonical(value):
    return json.dumps(value, sort_keys=True)


def run(config, dataset, output, method, seed):
    validate_method(config, method)
    train, _, _ = load_split(dataset, "train", config)
    validation, _, _ = load_split(dataset, "validation", config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source = provenance()
    manifest_hash = file_hash(Path(dataset)/"manifest.json")
    json_write(output/"request.json", {"config": config.to_dict(), "method": method,
        "seed": seed, "dataset_manifest_sha256": manifest_hash, "provenance": source})
    def progress(record):
        with (output/"progress.jsonl").open("a") as f:
            f.write(json.dumps(record, allow_nan=False)+"\n")
        print(f"{method} seed={seed} generation={record['generation']} "
              f"val_mse={record['best_validation_mse_rad2']:.6g}", flush=True)
    result = search(config, train, validation, method, seed, progress)
    result["dataset_counts"] = {k: v["count"] for k,v in
        json.loads((Path(dataset)/"manifest.json").read_text())["splits"].items()}
    save_result(output, config, result, manifest_hash, source)


def check_run(path, plan, method, seed, check_source=False):
    c, result, _ = load_result(path)
    if (canonical(c.to_dict()) != canonical(plan["config"]) or result["method"] != method
        or result["seed"] != seed or result["dataset_manifest_sha256"] != plan["dataset_manifest_sha256"]):
        raise ValueError(f"Existing run does not match requested protocol: {path}")
    if check_source:
        source = provenance()
        if any(result["provenance"][k] != source[k] for k in ("source_sha256", "packages")):
            raise ValueError("Source or packages changed. Start a new suite to avoid mixing implementations.")
    return result


def suite(args):
    c = Config.load(args.config)
    for method in args.methods:
        validate_method(c, method)
    if len(set(args.methods)) != len(args.methods) or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Duplicate methods or seeds")
    if any(not 0 <= s < 2**32 for s in args.seeds):
        raise ValueError("Seeds must be uint32 integers")
    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    dataset = Path(args.data).resolve() if args.data else root/"dataset"
    if not dataset.exists():
        if args.data:
            raise FileNotFoundError(dataset)
        generate(c, dataset)
    # Validate train/validation only. No test file is opened here.
    load_split(dataset, "train", c)
    load_split(dataset, "validation", c)
    plan = {"schema": 1, "config": c.to_dict(), "methods": args.methods, "seeds": args.seeds,
            "dataset": str(dataset), "dataset_manifest_sha256": file_hash(dataset/"manifest.json")}
    plan_path = root/"suite.json"
    if plan_path.exists() and canonical(json.loads(plan_path.read_text())) != canonical(plan):
        raise ValueError("Existing suite has a different protocol; choose a new --out")
    if (root/"frozen_protocol.json").exists():
        raise ValueError("Suite is frozen. Create a new suite for further optimization.")
    json_write(plan_path, plan)
    for method in args.methods:
        for seed in args.seeds:
            path = root/"runs"/f"{method}-seed-{seed}"
            if path.exists():
                check_run(path, plan, method, seed, check_source=True)
                print(f"Verified completed run: {path.name}", flush=True)
                continue
            partial = path.with_name(path.name+".partial")
            if partial.exists():
                interrupted = root/"interrupted"
                interrupted.mkdir(exist_ok=True)
                partial.rename(interrupted/f"{partial.name}-{time.time_ns()}")
            run(c, dataset, partial, method, seed)
            partial.rename(path)
    print(f"Search complete. Inspect validation and convergence, then: python -m qpi freeze --suite {root}")


def freeze(root):
    root = Path(root).resolve()
    plan = json.loads((root/"suite.json").read_text())
    destination = root/"frozen_protocol.json"
    if destination.exists():
        raise FileExistsError("Protocol already frozen; it will not be overwritten")
    if file_hash(Path(plan["dataset"])/"manifest.json") != plan["dataset_manifest_sha256"]:
        raise ValueError("Dataset manifest changed")
    runs = []
    for method in plan["methods"]:
        for seed in plan["seeds"]:
            path = root/"runs"/f"{method}-seed-{seed}"
            check_run(path, plan, method, seed, check_source=True)
            runs.append({"path": str(path.relative_to(root)), "result_sha256": file_hash(path/"result.json")})
    json_write(destination, {"schema": 1, "suite_sha256": file_hash(root/"suite.json"),
        "dataset_manifest_sha256": plan["dataset_manifest_sha256"], "runs": runs,
        "primary_endpoint": "held-out specimen expected phase RMSE at training photon budget, zero defocus",
        "secondary_endpoints": ["dose sweep", "unseen defocus", "shape OOD", "sampled camera error", "throughput"],
        "note": "Procedural freeze, not an access-control or preregistration service. Report exploratory test reuse."})
    print(f"Frozen {len(runs)} runs: {destination}")


def evaluate_suite(root):
    from .evaluate import evaluate_run
    root = Path(root).resolve()
    plan = json.loads((root/"suite.json").read_text())
    protocol = json.loads((root/"frozen_protocol.json").read_text())
    if file_hash(root/"suite.json") != protocol["suite_sha256"]:
        raise ValueError("Suite changed after freeze")
    if file_hash(Path(plan["dataset"])/"manifest.json") != protocol["dataset_manifest_sha256"]:
        raise ValueError("Dataset changed after freeze")
    for entry in protocol["runs"]:
        if file_hash(root/entry["path"]/"result.json") != entry["result_sha256"]:
            raise ValueError("A selected design changed after freeze")
    output = root/"evaluation"
    output.mkdir(exist_ok=True)
    request = {"protocol_sha256": file_hash(root/"frozen_protocol.json"), "provenance": provenance()}
    request_file = output/"request.json"
    if request_file.exists():
        previous = json.loads(request_file.read_text())
        if previous["protocol_sha256"] != request["protocol_sha256"] or any(
            previous["provenance"][k] != request["provenance"][k] for k in ("source_sha256", "packages")):
            raise ValueError("Evaluation code, dependencies or protocol changed")
    else:
        json_write(request_file, request)
    all_rows = []
    for entry in protocol["runs"]:
        run_path = root/entry["path"]
        path = output/run_path.name
        if not path.exists():
            partial = path.with_name(path.name+".partial")
            if partial.exists():
                partial.rename(path.with_name(path.name+f".interrupted-{time.time_ns()}"))
            evaluate_run(run_path, plan["dataset"], partial)
            partial.rename(path)
        metadata = json.loads((path/"evaluation.json").read_text())
        if metadata["result_sha256"] != entry["result_sha256"] or metadata["metrics_sha256"] != file_hash(path/"metrics.csv"):
            raise ValueError("Evaluation artifact checksum mismatch")
        with (path/"metrics.csv").open() as f:
            all_rows.extend(csv.DictReader(f))
        print(f"Evaluated {run_path.name}", flush=True)
    from .report import write_csv
    write_csv(output/"metrics.csv", all_rows)
    print(f"Metrics: {output/'metrics.csv'}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    data = sub.add_parser("data", help="Generate offline phase phantoms and split hashes")
    data.add_argument("--config", required=True)
    data.add_argument("--out", required=True)
    imported = sub.add_parser("import-data", help="Register pre-split calibrated phase-map NPZ files")
    imported.add_argument("--config", required=True)
    imported.add_argument("--source", required=True)
    imported.add_argument("--out", required=True)
    single = sub.add_parser("run", help="Train/select one method using train and validation only")
    single.add_argument("--config", required=True)
    single.add_argument("--data", required=True)
    single.add_argument("--method", choices=METHODS, required=True)
    single.add_argument("--seed", type=int, required=True)
    single.add_argument("--out", required=True)
    matrix = sub.add_parser("suite", help="Run a method/seed matrix; resume completed jobs")
    matrix.add_argument("--config", required=True)
    matrix.add_argument("--data")
    matrix.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    matrix.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    matrix.add_argument("--out", required=True)
    for name in ("freeze", "evaluate"):
        sub.add_parser(name).add_argument("--suite", required=True)
    check = sub.add_parser("check-grid", help="Fixed-design padding/pitch convergence on validation")
    check.add_argument("--run", required=True)
    check.add_argument("--data", required=True)
    check.add_argument("--out", required=True)
    summary = sub.add_parser("report", help="Export measured tables and dose/accuracy plot")
    summary.add_argument("--metrics", required=True)
    summary.add_argument("--out", required=True)
    args = p.parse_args()
    if args.command == "data":
        generate(Config.load(args.config), args.out)
    elif args.command == "import-data":
        import_directory(Config.load(args.config), args.source, args.out)
    elif args.command == "run":
        run(Config.load(args.config), args.data, args.out, args.method, args.seed)
    elif args.command == "suite":
        suite(args)
    elif args.command == "freeze":
        freeze(args.suite)
    elif args.command == "evaluate":
        evaluate_suite(args.suite)
    elif args.command == "check-grid":
        from .evaluate import grid_check
        print(json.dumps(grid_check(args.run, args.data, args.out), indent=2))
    elif args.command == "report":
        from .report import report
        report(args.metrics, args.out)


if __name__ == "__main__":
    main()

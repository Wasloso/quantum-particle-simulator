import argparse
from contextlib import suppress
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from lih_vqe.config import load_config
from lih_vqe.enums import RunStatus


def print_run(run: dict) -> None:
    distance = run["config"]["molecule"]["distance"]
    if run["status"] == RunStatus.FAILED:
        print(f"R={distance:g} Å: FAILED: {run['error']}", flush=True)
        return
    result = run["vqe"]
    print(
        f"R={distance:g} Å | {run['model']['num_qubits']} qubits | "
        f"VQE={result['energy']:.10f} Eh | "
        f"error vs CASCI={run['errors_millihartree']['active']:.4f} mHa | "
        f"model error={run['errors_millihartree']['model_reduction']:.4f} mHa",
        flush=True,
    )
    print(
        f"  {len(result['history'])} objective calls; converged={result['converged']}; "
        f"sector_valid={run['sector_valid']}: {result['message']}",
        flush=True,
    )
    if "noisy_energy_hartree" in run["evaluation"]:
        print(f"  Noisy expectation: {run['evaluation']['noisy_energy_hartree']:.10f} Eh")
    if "sampled" in run["evaluation"]:
        sample = run["evaluation"]["sampled"]
        print(
            f"  Sampled: {sample['energy_hartree']:.10f} ± "
            f"{sample['standard_error_hartree']:.6f} Eh (standard error), "
            f"{sample['shots_used']} shots across {sample['groups']} groups"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Small LiH VQE experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    for command, help_text in [
        ("run", "one bond length"),
        ("scan", "a bond-length scan"),
        ("models", "compare the 12-qubit reference and reduced LiH models without VQE"),
    ]:
        subparser = commands.add_parser(command, help=help_text)
        subparser.add_argument("--config", type=Path, default=Path("configs/lih.toml"))
        subparser.add_argument("--output", type=Path, help="new JSON file (default: timestamped)")
    plot = commands.add_parser("plot", help="plot saved JSON results")
    plot.add_argument("results", nargs="+", type=Path)
    plot.add_argument("--output", type=Path, default=Path("results/plots"))
    args = parser.parse_args(argv)
    results, output = None, None
    try:
        if args.command == "plot":
            from lih_vqe.plotting import plot_results

            for path in plot_results(args.results, args.output):
                print(path)
            return 0
        from lih_vqe.experiments import (
            compare_models,
            new_results,
            run_experiment,
            save_results,
            scan,
        )

        config = load_config(args.config)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        output = args.output or Path("results") / f"{args.command}-{stamp}-{uuid4().hex[:6]}.json"
        if output.exists():
            raise ValueError(f"output already exists: {output}; choose a new filename")
        results = new_results(args.command)
        results["config"] = asdict(config)
        save_results(output, results)
        print(f"Starting {args.command}; saving to {output}", flush=True)
        if args.command == "models":
            results["models"] = []
            print(
                f"{'Model':<26} {'e-':>3} {'MO':>3} {'Qubits':>7} {'Error vs full FCI (mHa)':>24}"
            )
            for row in compare_models(config):
                results["models"].append(row)
                save_results(output, results)
                model = row["model"]
                print(
                    f"{row['label']:<26} {sum(model['active_particles']):>3} "
                    f"{len(model['active_orbitals']):>3} {model['num_qubits']:>7} "
                    f"{row['model_error_millihartree']:>24.6f}",
                    flush=True,
                )
        else:
            runs = [run_experiment(config)] if args.command == "run" else scan(config)
            for run in runs:
                results["runs"].append(run)
                save_results(output, results)
                print_run(run)
        failed = any(run["status"] == RunStatus.FAILED for run in results["runs"])
        results["status"] = RunStatus.COMPLETED_WITH_FAILURES if failed else RunStatus.COMPLETED
        save_results(output, results)
        return int(failed)
    except (OSError, ValueError, RuntimeError) as error:
        if results is not None:
            results["status"] = RunStatus.FAILED
            results["error"] = str(error)
            with suppress(OSError):
                save_results(output, results)
        parser.exit(1, f"Error: {error}\n")
    except KeyboardInterrupt:
        if results is not None:
            results["status"] = RunStatus.INTERRUPTED
            save_results(output, results)
        parser.exit(130, "Interrupted; completed geometries remain in the result file.\n")


if __name__ == "__main__":
    raise SystemExit(main())

import json
import platform
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from lih_vqe.ansatz import build_ansatz
from lih_vqe.chemistry import build_problem
from lih_vqe.config import ExperimentConfig
from lih_vqe.enums import RunStatus
from lih_vqe.evaluation import noisy_energy, sampled_energy
from lih_vqe.vqe import optimize


def run_experiment(config: ExperimentConfig) -> dict:
    start = perf_counter()
    problem = build_problem(config)
    ansatz = build_ansatz(problem, config.vqe)
    result = optimize(problem, ansatz, config.vqe)
    circuit = ansatz.circuit.assign_parameters(result.parameters)
    evaluation = {}
    if config.evaluation.noise:
        evaluation["noisy_energy_hartree"] = noisy_energy(
            circuit, problem.hamiltonian, problem.energy_offset, config.evaluation.noise
        )
    if config.evaluation.shots:
        evaluation["sampled"] = sampled_energy(
            circuit,
            problem.hamiltonian,
            problem.energy_offset,
            config.evaluation.shots,
            config.evaluation.seed,
            config.evaluation.noise,
        )
    references = problem.references
    return {
        "status": RunStatus.COMPLETED,
        "config": asdict(config),
        "model": problem.metadata,
        "references_hartree": references,
        "vqe": asdict(result),
        "sector_valid": abs(result.diagnostics["constraint_violation"]) < 1e-6,
        "errors_millihartree": {
            "active": 1000 * (result.energy - references["active_casci"]),
            "full": 1000 * (result.energy - references["full_fci"]),
            "model_reduction": 1000 * (references["active_casci"] - references["full_fci"]),
        },
        "resources": ansatz.resources,
        "parameter_order": [str(parameter) for parameter in ansatz.circuit.parameters],
        "compilation": {"basis_gates": ["rz", "sx", "x", "cx"], "connectivity": "all-to-all"},
        "hamiltonian": {
            "pauli_terms": [
                {"pauli": label, "coefficient": float(coefficient.real)}
                for label, coefficient in problem.hamiltonian.to_list()
            ],
            "energy_offset_hartree": problem.energy_offset,
        },
        "evaluation": evaluation,
        "elapsed_seconds": perf_counter() - start,
    }


def new_results(kind: str) -> dict:
    return {
        "schema_version": 1,
        "kind": kind,
        "status": RunStatus.RUNNING,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "units": {"distance": "angstrom", "energy": "hartree", "error": "millihartree"},
        "software": {
            "python": platform.python_version(),
            **{
                name: version(name)
                for name in [
                    "lih-vqe",
                    "numpy",
                    "qiskit",
                    "qiskit-nature",
                    "qiskit-aer",
                    "pyscf",
                    "scipy",
                ]
            },
        },
        "runs": [],
    }


def save_results(path: str | Path, results: dict) -> None:
    path = Path(path)
    serialized = json.dumps(results, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(path)

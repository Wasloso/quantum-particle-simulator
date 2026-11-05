from dataclasses import dataclass

import numpy as np
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize

from lih_vqe.ansatz import Ansatz
from lih_vqe.chemistry import MolecularProblem
from lih_vqe.config import VQEConfig
from lih_vqe.enums import AnsatzKind, Optimizer


@dataclass
class VQEResult:
    parameters: list[float]
    energy: float
    objective: float
    penalty_strength: float
    diagnostics: dict[str, float]
    history: list[dict[str, float | int]]
    converged: bool
    message: str


class EvaluationLimit(Exception):
    pass


def optimize(problem: MolecularProblem, ansatz: Ansatz, config: VQEConfig) -> VQEResult:
    hamiltonian = problem.hamiltonian.to_matrix(sparse=True)
    observables = {name: op.to_matrix(sparse=True) for name, op in problem.observables.items()}
    constraint = observables["constraint_violation"]
    # The penalty must exceed the spectral width, excluding constant energy shifts.
    nonconstant = [
        abs(coefficient)
        for pauli, coefficient in problem.hamiltonian.to_list()
        if set(pauli) != {"I"}
    ]
    strength = float(2 * sum(nonconstant) + 1)
    history = []
    best_value = float("inf")
    best_parameters = None

    def evaluate(parameters: np.ndarray) -> float:
        nonlocal best_value, best_parameters
        if len(history) >= config.max_evaluations:
            raise EvaluationLimit
        state = Statevector.from_instruction(ansatz.circuit.assign_parameters(parameters)).data
        energy = float(np.vdot(state, hamiltonian @ state).real) + problem.energy_offset
        violation = float(np.vdot(state, constraint @ state).real)
        objective = energy + strength * violation
        history.append({"evaluation": len(history) + 1, "energy": energy, "objective": objective})
        if objective < best_value:
            best_value, best_parameters = objective, np.array(parameters, copy=True)
        return objective

    rng = np.random.default_rng(config.seed)
    initial = (
        np.zeros(ansatz.circuit.num_parameters)
        if config.ansatz == AnsatzKind.UCCSD
        else rng.normal(0, 0.1, ansatz.circuit.num_parameters)
    )
    options = {"maxiter": config.max_evaluations}
    if config.optimizer == Optimizer.SLSQP:
        options["ftol"] = config.tolerance
    try:
        result = minimize(
            evaluate, initial, method=config.optimizer.value, tol=config.tolerance, options=options
        )
        converged, message = bool(result.success), str(result.message)
    except EvaluationLimit:
        converged, message = False, "objective evaluation budget exhausted"
    # Choose by penalized objective: a low physical energy can belong to the wrong sector.
    state = Statevector.from_instruction(ansatz.circuit.assign_parameters(best_parameters)).data
    diagnostics = {
        name: float(np.vdot(state, matrix @ state).real) for name, matrix in observables.items()
    }
    energy = float(np.vdot(state, hamiltonian @ state).real) + problem.energy_offset
    return VQEResult(
        parameters=best_parameters.tolist(),
        energy=energy,
        objective=energy + strength * diagnostics["constraint_violation"],
        penalty_strength=strength,
        diagnostics=diagnostics,
        history=history,
        converged=converged,
        message=message,
    )
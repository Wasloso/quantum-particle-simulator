import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error


def make_simulator(noise: float) -> AerSimulator:
    if not np.isfinite(noise) or not 0 <= noise <= 1:
        raise ValueError("noise must be between zero and one")
    model = NoiseModel()
    if noise:
        model.add_all_qubit_quantum_error(depolarizing_error(noise, 2), ["cx"])
    return AerSimulator(method="density_matrix", noise_model=model, max_parallel_threads=1)


def measurement_circuit(circuit: QuantumCircuit, group: SparsePauliOp) -> QuantumCircuit:
    measured = circuit.copy()
    labels = [label for label, _ in group.to_list()]
    for qubit in range(circuit.num_qubits):
        axes = {label[-1 - qubit] for label in labels} - {"I"}
        if len(axes) > 1:
            raise ValueError("measurement group is not qubit-wise commuting")
        axis = next(iter(axes), "I")
        if axis == "Y":
            measured.sdg(qubit)
        if axis in {"X", "Y"}:
            measured.h(qubit)
    measured.measure_all()
    return transpile(measured, basis_gates=["rz", "sx", "x", "cx"], optimization_level=0)


def group_moments(counts: dict[str, int], group: SparsePauliOp) -> tuple[float, float]:
    """Return the group mean and variance of that mean, including term covariances."""
    shots = sum(counts.values())
    if shots < 2:
        raise ValueError("at least two shots are needed to estimate a variance")
    terms = [
        (int("".join("0" if axis == "I" else "1" for axis in label), 2), coefficient.real)
        for label, coefficient in group.to_list()
    ]
    weighted_sum, weighted_squares = 0.0, 0.0
    for bitstring, count in counts.items():
        bits = int(bitstring.replace(" ", ""), 2)
        value = sum(c * (-1) ** ((bits & mask).bit_count()) for mask, c in terms)
        weighted_sum += count * value
        weighted_squares += count * value * value
    mean = weighted_sum / shots
    sample_variance = max(0.0, (weighted_squares - shots * mean * mean) / (shots - 1))
    return float(mean), float(sample_variance / shots)


def sampled_energy(
    circuit: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    offset: float,
    shots: int,
    seed: int,
    noise: float = 0.0,
) -> dict:
    """Split a total shot budget across qubit-wise commuting measurement groups."""
    if type(shots) is not int or shots <= 0:
        raise ValueError("shots must be a positive integer")
    constant = offset
    terms = []
    for label, coefficient in hamiltonian.to_list():
        if abs(coefficient.imag) > 1e-10:
            raise ValueError("Hamiltonian coefficients must be real")
        if set(label) == {"I"}:
            constant += coefficient.real
        elif abs(coefficient) > 1e-12:
            terms.append((label, coefficient))
    groups = SparsePauliOp.from_list(terms).group_commuting(qubit_wise=True) if terms else []
    if shots < 2 * len(groups):
        raise ValueError(f"need at least {2 * len(groups)} total shots for {len(groups)} groups")
    simulator = make_simulator(noise)
    mean, variance = float(constant), 0.0
    allocation = []
    for index, group in enumerate(groups):
        group_shots = shots // len(groups) + (index < shots % len(groups))
        measured = measurement_circuit(circuit, group)
        counts = (
            simulator.run(measured, shots=group_shots, seed_simulator=seed + index)
            .result()
            .get_counts()
        )
        group_mean, group_variance = group_moments(counts, group)
        mean += group_mean
        variance += group_variance
        allocation.append(group_shots)
    return {
        "energy_hartree": float(mean),
        "standard_error_hartree": float(np.sqrt(variance)),
        "shots_requested": shots,
        "shots_used": sum(allocation),
        "groups": len(groups),
        "shots_per_group": allocation,
        "noise": noise,
        "seed": seed,
    }

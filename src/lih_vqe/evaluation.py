from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp


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

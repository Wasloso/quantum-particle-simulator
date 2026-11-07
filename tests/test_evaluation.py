import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from lih_vqe.evaluation import sampled_energy


def test_sampling_y_basis_and_identity_offset():
    circuit = QuantumCircuit(1)
    circuit.h(0)
    circuit.s(0)  # +1 eigenstate of Y
    hamiltonian = SparsePauliOp.from_list([("Y", 2), ("I", 3)])
    result = sampled_energy(circuit, hamiltonian, offset=4, shots=200, seed=7)
    assert result["energy_hartree"] == pytest.approx(9)
    assert result["standard_error_hartree"] == pytest.approx(0)
    assert result["shots_used"] == 200


def test_bit_order_uses_qiskit_little_endian_convention():
    circuit = QuantumCircuit(2)
    circuit.x(0)  # |01>: IZ = -1, ZI = +1
    hamiltonian = SparsePauliOp.from_list([("IZ", 2), ("ZI", 3)])
    result = sampled_energy(circuit, hamiltonian, offset=0, shots=200, seed=7)
    assert result["energy_hartree"] == pytest.approx(1)


def test_shots_are_divided_between_groups_and_reproducible():
    circuit = QuantumCircuit(1)
    hamiltonian = SparsePauliOp.from_list([("X", 1), ("Z", 1)])
    first = sampled_energy(circuit, hamiltonian, offset=0, shots=201, seed=5)
    second = sampled_energy(circuit, hamiltonian, offset=0, shots=201, seed=5)
    assert first == second
    assert first["shots_used"] == 201
    assert sorted(first["shots_per_group"]) == [100, 101]
    assert abs(first["energy_hartree"] - 1) < 5 * first["standard_error_hartree"]


def test_rejects_shot_budget_too_small_for_variance():
    with pytest.raises(ValueError, match="at least"):
        sampled_energy(QuantumCircuit(1), SparsePauliOp.from_list([("X", 1), ("Z", 1)]), 0, 2, 5)

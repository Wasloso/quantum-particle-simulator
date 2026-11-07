import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from lih_vqe.evaluation import group_moments


def test_group_variance_includes_covariance():
    # Both Z outcomes are perfectly correlated: each shot's energy is +2 or -2.
    # Treating the two terms as independent would underestimate the uncertainty.
    group = SparsePauliOp.from_list([("ZI", 1), ("IZ", 1)])
    mean, variance = group_moments({"00": 50, "11": 50}, group)
    assert mean == 0
    assert variance == pytest.approx(4 / 99)


def test_rejects_incompatible_axes_on_one_qubit():
    from lih_vqe.evaluation import measurement_circuit

    group = SparsePauliOp.from_list([("X", 1), ("Y", 1)])
    with pytest.raises(ValueError, match="qubit-wise"):
        measurement_circuit(QuantumCircuit(1), group)

import pytest
from qiskit.quantum_info import SparsePauliOp

from lih_vqe.chemistry import sector_reference


def test_sector_reference_does_not_choose_unphysical_global_minimum():
    # |0> has energy +1 and satisfies the constraint. |1> has lower energy -1
    # but violates the constraint. The correct restricted reference is +1.
    hamiltonian = SparsePauliOp.from_list([("Z", 1)])
    constraint = SparsePauliOp.from_list([("I", 0.5), ("Z", -0.5)])
    energy, dimension = sector_reference(hamiltonian, constraint)
    assert energy == pytest.approx(1)
    assert dimension == 1


def test_reference_filters_alpha_and_beta_occupations():
    hamiltonian = SparsePauliOp.from_list([("ZI", 2), ("IZ", 1)])
    constraint = SparsePauliOp.from_list([("II", 0)])
    alpha = SparsePauliOp.from_list([("II", 0.5), ("IZ", -0.5)])
    beta = SparsePauliOp.from_list([("II", 0.5), ("ZI", -0.5)])
    energy, dimension = sector_reference(hamiltonian, constraint, (alpha, beta), (1, 0))
    assert energy == pytest.approx(1)
    assert dimension == 1


def test_reference_rejects_an_empty_physical_sector():
    hamiltonian = SparsePauliOp.from_list([("Z", 1)])
    constraint = SparsePauliOp.from_list([("I", 1)])
    with pytest.raises(ValueError, match="no states"):
        sector_reference(hamiltonian, constraint)

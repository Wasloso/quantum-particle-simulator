import numpy as np
from qiskit.quantum_info import SparsePauliOp


def sector_reference(
    hamiltonian: SparsePauliOp,
    constraint: SparsePauliOp,
    number_operators: tuple[SparsePauliOp, SparsePauliOp] | None = None,
    particles: tuple[int, int] | None = None,
) -> tuple[float, int]:
    """Restrict to the requested particle and singlet sector before diagonalization."""
    indices = np.arange(2**hamiltonian.num_qubits)
    if number_operators is not None:
        if particles is None:
            raise ValueError("particle counts are required with number operators")
        valid = np.ones(len(indices), dtype=bool)
        for operator, target in zip(number_operators, particles, strict=True):
            occupations = operator.to_matrix(sparse=True).diagonal().real
            valid &= np.isclose(occupations, target, atol=1e-8, rtol=0)
        indices = indices[valid]
    if len(indices) > 1000:
        raise ValueError("reference sector is too large for this small-system implementation")
    constraint_block = constraint.to_matrix(sparse=True)[indices][:, indices].toarray()
    values, vectors = np.linalg.eigh(constraint_block)
    basis = vectors[:, np.abs(values) < 1e-8]
    if basis.shape[1] == 0:
        raise ValueError("the mapped Hamiltonian has no states in the requested sector")
    block = hamiltonian.to_matrix(sparse=True)[indices][:, indices].toarray()
    restricted = basis.conj().T @ block @ basis
    return float(np.linalg.eigvalsh(restricted)[0]), basis.shape[1]

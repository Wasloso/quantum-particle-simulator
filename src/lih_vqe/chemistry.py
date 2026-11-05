from dataclasses import dataclass

import numpy as np
from pyscf import fci, gto, lib, mcscf, scf
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper, QubitMapper
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.properties import AngularMomentum
from qiskit_nature.units import DistanceUnit

from lih_vqe.config import ExperimentConfig
from lih_vqe.enums import MappingKind
from lih_vqe.models import make_model


@dataclass
class MolecularProblem:
    problem: ElectronicStructureProblem
    mapper: QubitMapper
    hamiltonian: SparsePauliOp
    observables: dict[str, SparsePauliOp]
    energy_offset: float
    references: dict[str, float]
    metadata: dict


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


def build_problem(config: ExperimentConfig) -> MolecularProblem:
    config.validate()
    spec, model = config.molecule, config.model
    lib.num_threads(spec.threads)
    atoms = f"{spec.symbols[0]} 0 0 0; {spec.symbols[1]} 0 0 {spec.distance}"
    mol = gto.M(atom=atoms, basis=spec.basis, unit="Angstrom", charge=0, spin=0, verbose=0)
    if mol.nelectron % 2:
        raise ValueError("only neutral, closed-shell molecules are supported")
    mean_field = scf.RHF(mol)
    mean_field.conv_tol = 1e-10
    mean_field.kernel()
    if not mean_field.converged:
        raise RuntimeError("RHF did not converge; the molecular reference is unreliable")
    n_orbitals = mean_field.mo_coeff.shape[1]
    full_problem = PySCFDriver(
        atom=atoms, basis=spec.basis, unit=DistanceUnit.ANGSTROM, spin=0, conv_tol=1e-10
    ).run()
    problem, selection = make_model(model).apply(full_problem)

    solver = fci.FCI(mean_field)
    full_energy, full_ci = solver.kernel()
    if not solver.converged:
        raise RuntimeError("full-space FCI did not converge")
    full_spin = float(solver.spin_square(full_ci, n_orbitals, mol.nelec)[0])
    cas = mcscf.CASCI(mean_field, len(selection.orbitals), selection.electrons)
    cas.verbose = 0
    cas.kernel(cas.sort_mo(list(selection.orbitals), base=0))
    if not cas.fcisolver.converged:
        raise RuntimeError("active-space CASCI did not converge")
    cas_spin = float(cas.fcisolver.spin_square(cas.ci, cas.ncas, cas.nelecas)[0])
    if abs(full_spin) > 1e-6 or abs(cas_spin) > 1e-6:
        raise RuntimeError("the classical ground state is not the expected singlet")

    particles = problem.num_particles
    n = problem.num_spatial_orbitals
    mapper = (
        ParityMapper(num_particles=particles if model.two_qubit_reduction else None)
        if model.mapping == MappingKind.PARITY
        else JordanWignerMapper()
    )
    hamiltonian = mapper.map(problem.hamiltonian.second_q_op()).simplify()
    identity = SparsePauliOp.from_list([("I" * hamiltonian.num_qubits, 1.0)])
    alpha = mapper.map(
        FermionicOp({f"+_{i} -_{i}": 1.0 for i in range(n)}, num_spin_orbitals=2 * n)
    )
    beta = mapper.map(
        FermionicOp({f"+_{i} -_{i}": 1.0 for i in range(n, 2 * n)}, num_spin_orbitals=2 * n)
    )
    spin = mapper.map(AngularMomentum(n).second_q_ops()["AngularMomentum"])
    delta_alpha, delta_beta = alpha - particles[0] * identity, beta - particles[1] * identity
    # This nonnegative constraint is zero only in the target electron/singlet sector.
    constraint = (delta_alpha @ delta_alpha + delta_beta @ delta_beta + spin).simplify()
    offset = float(sum(problem.hamiltonian.constants.values()))
    active_reference, sector_dimension = sector_reference(
        hamiltonian, constraint, (alpha, beta), particles
    )
    active_reference += offset
    if not np.isclose(active_reference, cas.e_tot, atol=1e-8, rtol=0):
        raise RuntimeError("mapped singlet energy disagrees with PySCF CASCI")

    hf = HartreeFock(n, particles, mapper)
    hf_energy = float(Statevector.from_instruction(hf).expectation_value(hamiltonian).real) + offset
    if not np.isclose(hf_energy, mean_field.e_tot, atol=1e-8, rtol=0):
        raise ValueError("active orbitals must preserve the RHF reference occupations")
    return MolecularProblem(
        problem=problem,
        mapper=mapper,
        hamiltonian=hamiltonian,
        observables={
            "alpha_electrons": alpha,
            "beta_electrons": beta,
            "spin_squared": spin,
            "constraint_violation": constraint,
        },
        energy_offset=offset,
        references={
            "hartree_fock": float(mean_field.e_tot),
            "full_fci": float(full_energy),
            "active_casci": float(cas.e_tot),
            "mapped_sector": active_reference,
        },
        metadata={
            "kind": selection.name,
            "num_qubits": hamiltonian.num_qubits,
            "qubits_before_symmetry_reduction": 2 * n,
            "full_space_qubits": 2 * n_orbitals,
            "num_pauli_terms": len(hamiltonian),
            "sector_dimension": sector_dimension,
            "active_particles": list(particles),
            "active_orbitals": list(selection.orbitals),
            "orbital_energies_hartree": mean_field.mo_energy.tolist(),
            "energy_constants_hartree": {
                key: float(value) for key, value in problem.hamiltonian.constants.items()
            },
            "full_fci_spin_squared": full_spin,
            "active_casci_spin_squared": cas_spin,
        },
    )

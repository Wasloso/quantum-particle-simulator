from dataclasses import dataclass

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock

from lih_vqe.chemistry import MolecularProblem
from lih_vqe.config import VQEConfig
from lih_vqe.enums import AnsatzKind

@dataclass
class Ansatz:
    circuit: QuantumCircuit
    resources: dict[str, int]

def build_ansatz(problem: MolecularProblem, config: VQEConfig) -> Ansatz:
    electronic = problem.problem
    hf = HartreeFock(electronic.num_spatial_orbitals, electronic.num_particles, problem.mapper)
    if config.ansatz == AnsatzKind.UCCSD:
        circuit = UCCSD(
            electronic.num_spatial_orbitals,
            electronic.num_particles,
            problem.mapper,
            initial_state=hf,
            reps=config.reps,
            preserve_spin=True,
        )
    elif config.ansatz == AnsatzKind.HEA:
        circuit = QuantumCircuit(hf.num_qubits)
        circuit.compose(hf, inplace=True)
        parameters = iter(ParameterVector("theta", 2 * circuit.num_qubits * config.reps))
        for _ in range(config.reps):
            for qubit in range(circuit.num_qubits):
                circuit.ry(next(parameters), qubit)
                circuit.rz(next(parameters), qubit)
            for qubit in range(circuit.num_qubits - 1):
                circuit.cx(qubit, qubit + 1)
    else:
        raise ValueError(f"unknown ansatz: {config.ansatz}")
    compiled = transpile(
        circuit,
        basis_gates=["rz", "sx", "x", "cx"],
        optimization_level=1,
        seed_transpiler=config.seed,
    )
    return Ansatz(
        compiled,
        {
            "qubits": compiled.num_qubits,
            "parameters": compiled.num_parameters,
            "depth": compiled.depth(),
            "cx_gates": int(compiled.count_ops().get("cx", 0)),
            "total_gates": compiled.size(),
        },
    )
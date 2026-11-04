import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from lih_vqe.enums import AnsatzKind, MappingKind, ModelKind, Optimizer


@dataclass(frozen=True)
class MoleculeConfig:
    symbols: tuple[str, str] = ("Li", "H")
    distance: float = 1.6  # Angstrom
    basis: str = "sto-3g"
    threads: int = 1


@dataclass(frozen=True)
class ModelConfig:
    kind: ModelKind = ModelKind.ACTIVE
    active_electrons: int = 2
    active_orbitals: tuple[int, ...] = (1, 2, 3, 5)
    mapping: MappingKind = MappingKind.PARITY
    two_qubit_reduction: bool = True

    def __post_init__(self):
        object.__setattr__(self, "kind", ModelKind(self.kind))
        object.__setattr__(self, "mapping", MappingKind(self.mapping))


@dataclass(frozen=True)
class VQEConfig:
    ansatz: AnsatzKind = AnsatzKind.UCCSD
    reps: int = 1
    optimizer: Optimizer = Optimizer.SLSQP
    max_evaluations: int = 500
    tolerance: float = 1e-8
    seed: int = 42

    def __post_init__(self):
        object.__setattr__(self, "ansatz", AnsatzKind(self.ansatz))
        object.__setattr__(self, "optimizer", Optimizer(self.optimizer))


@dataclass(frozen=True)
class EvaluationConfig:
    shots: int = 0  # Total budget across measurement groups; zero disables sampling
    noise: float = 0.0  # Depolarizing parameter after each compiled CX
    seed: int = 123


@dataclass(frozen=True)
class ScanConfig:
    distances: tuple[float, ...] = (1.2, 1.6, 2.0, 2.5, 3.1)


@dataclass(frozen=True)
class ExperimentConfig:
    molecule: MoleculeConfig = field(default_factory=MoleculeConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    vqe: VQEConfig = field(default_factory=VQEConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    scan: ScanConfig = field(default_factory=ScanConfig)

    def validate(self) -> None:
        if len(self.molecule.symbols) != 2 or not all(self.molecule.symbols):
            raise ValueError("molecule.symbols must contain two element symbols")
        if not self.molecule.basis.strip():
            raise ValueError("molecule.basis must not be empty")
        if not self.scan.distances:
            raise ValueError("scan.distances must not be empty")
        if any(
            not math.isfinite(distance) or distance <= 0
            for distance in (self.molecule.distance, *self.scan.distances)
        ):
            raise ValueError("distances must be finite, positive values in Angstrom")
        if self.model.kind == ModelKind.ACTIVE:
            orbitals = self.model.active_orbitals
            if not orbitals or any(type(i) is not int or i < 0 for i in orbitals):
                raise ValueError("model.active_orbitals must contain nonnegative integer indices")
            if tuple(sorted(set(orbitals))) != tuple(orbitals):
                raise ValueError("model.active_orbitals must be sorted and unique")
            electrons = self.model.active_electrons
            if type(electrons) is not int or electrons < 2 or electrons % 2:
                raise ValueError("model.active_electrons must be a positive even integer")
            if electrons >= 2 * len(orbitals):
                raise ValueError("the active space needs occupied and unoccupied spin orbitals")
        if type(self.model.two_qubit_reduction) is not bool:
            raise ValueError("model.two_qubit_reduction must be a boolean")
        if self.model.two_qubit_reduction and self.model.mapping != MappingKind.PARITY:
            raise ValueError("two_qubit_reduction is only supported with parity mapping")
        for name, value, minimum in [
            ("molecule.threads", self.molecule.threads, 1),
            ("vqe.reps", self.vqe.reps, 1),
            ("vqe.max_evaluations", self.vqe.max_evaluations, 1),
            ("vqe.seed", self.vqe.seed, 0),
            ("evaluation.seed", self.evaluation.seed, 0),
            ("evaluation.shots", self.evaluation.shots, 0),
        ]:
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not math.isfinite(self.vqe.tolerance) or self.vqe.tolerance <= 0:
            raise ValueError("vqe.tolerance must be finite and positive")
        if not math.isfinite(self.evaluation.noise) or not 0 <= self.evaluation.noise <= 1:
            raise ValueError("evaluation.noise must be between 0 and 1")


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("rb") as stream:
        data = tomllib.load(stream)
    unknown = set(data) - {"molecule", "model", "vqe", "evaluation", "scan"}
    if unknown:
        raise ValueError(f"unknown configuration sections: {', '.join(sorted(unknown))}")
    try:
        molecule = dict(data.get("molecule", {}))
        model = dict(data.get("model", {}))
        scan = dict(data.get("scan", {}))
        if "symbols" in molecule:
            molecule["symbols"] = tuple(molecule["symbols"])
        if "active_orbitals" in model:
            model["active_orbitals"] = tuple(model["active_orbitals"])
        if "distances" in scan:
            scan["distances"] = tuple(scan["distances"])
        config = ExperimentConfig(
            molecule=MoleculeConfig(**molecule),
            model=ModelConfig(**model),
            vqe=VQEConfig(**data.get("vqe", {})),
            evaluation=EvaluationConfig(**data.get("evaluation", {})),
            scan=ScanConfig(**scan),
        )
        config.validate()
        return config
    except (TypeError, AttributeError) as error:
        raise ValueError(f"invalid configuration: {error}") from error

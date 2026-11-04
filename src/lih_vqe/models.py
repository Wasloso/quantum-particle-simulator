from abc import ABC, abstractmethod
from dataclasses import dataclass

from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer, FreezeCoreTransformer

from lih_vqe.config import ModelConfig
from lih_vqe.enums import ModelKind


@dataclass(frozen=True)
class SpaceSelection:
    name: ModelKind
    electrons: int
    orbitals: tuple[int, ...]


class ElectronicModel(ABC):
    @abstractmethod
    def select(self, problem: ElectronicStructureProblem) -> SpaceSelection:
        pass

    def apply(
        self, problem: ElectronicStructureProblem
    ) -> tuple[ElectronicStructureProblem, SpaceSelection]:
        selection = self.select(problem)
        if selection.electrons < 2 or selection.electrons % 2:
            raise ValueError("the selected model must contain a positive even electron count")
        if selection.electrons > sum(problem.num_particles):
            raise ValueError("active_electrons exceeds the molecule's electron count")
        if not selection.orbitals or max(selection.orbitals) >= problem.num_spatial_orbitals:
            raise ValueError("selected orbital index exceeds the molecule's spatial orbitals")
        if selection.electrons >= 2 * len(selection.orbitals):
            raise ValueError("the selected model needs unoccupied spin orbitals for VQE")
        if selection.electrons == sum(problem.num_particles) and selection.orbitals == tuple(
            range(problem.num_spatial_orbitals)
        ):
            return problem, selection
        reduced = ActiveSpaceTransformer(
            selection.electrons,
            len(selection.orbitals),
            active_orbitals=list(selection.orbitals),
        ).transform(problem)
        return reduced, selection


class FullSpace(ElectronicModel):
    def select(self, problem: ElectronicStructureProblem) -> SpaceSelection:
        return SpaceSelection(
            ModelKind.FULL, sum(problem.num_particles), tuple(range(problem.num_spatial_orbitals))
        )


class FrozenCore(ElectronicModel):
    def select(self, problem: ElectronicStructureProblem) -> SpaceSelection:
        core = FreezeCoreTransformer().count_core_orbitals(problem.molecule.symbols)
        return SpaceSelection(
            ModelKind.FROZEN_CORE,
            sum(problem.num_particles) - 2 * core,
            tuple(range(core, problem.num_spatial_orbitals)),
        )


@dataclass
class ActiveSpace(ElectronicModel):
    electrons: int
    orbitals: tuple[int, ...]

    def select(self, problem: ElectronicStructureProblem) -> SpaceSelection:
        return SpaceSelection(ModelKind.ACTIVE, self.electrons, self.orbitals)


def make_model(config: ModelConfig) -> ElectronicModel:
    if config.kind == ModelKind.FULL:
        return FullSpace()
    if config.kind == ModelKind.FROZEN_CORE:
        return FrozenCore()
    if config.kind == ModelKind.ACTIVE:
        return ActiveSpace(config.active_electrons, config.active_orbitals)
    raise ValueError(f"unknown molecular model: {config.kind}")

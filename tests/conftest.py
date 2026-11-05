from dataclasses import replace

import pytest

from lih_vqe.chemistry import build_problem
from lih_vqe.config import ExperimentConfig, ModelConfig, MoleculeConfig


@pytest.fixture(scope="session")
def lih_config():
    return replace(ExperimentConfig(), model=ModelConfig(active_orbitals=(1, 2, 5)))


@pytest.fixture(scope="session")
def lih_problem(lih_config):
    return build_problem(lih_config)


@pytest.fixture(scope="session")
def hydrogen_config():
    return replace(
        ExperimentConfig(),
        molecule=MoleculeConfig(symbols=("H", "H"), distance=0.735),
        model=ModelConfig(active_orbitals=(0, 1)),
    )


@pytest.fixture(scope="session")
def hydrogen_problem(hydrogen_config):
    return build_problem(hydrogen_config)

import pytest
from pyscf import lib
from qiskit_nature.second_q.drivers import PySCFDriver

from lih_vqe.config import load_config
from lih_vqe.enums import ModelKind
from lih_vqe.models import ActiveSpace, FrozenCore, FullSpace, make_model


@pytest.fixture(scope="module")
def molecule():
    lib.num_threads(1)
    return PySCFDriver(atom="Li 0 0 0; H 0 0 1.6", basis="sto3g").run()


@pytest.mark.parametrize(
    ("model", "kind", "electrons", "orbitals"),
    [
        (FullSpace(), ModelKind.FULL, 4, (0, 1, 2, 3, 4, 5)),
        (FrozenCore(), ModelKind.FROZEN_CORE, 2, (1, 2, 3, 4, 5)),
        (ActiveSpace(2, (1, 2, 3, 5)), ModelKind.ACTIVE, 2, (1, 2, 3, 5)),
    ],
)
def test_select_and_apply_model(molecule, model, kind, electrons, orbitals):
    reduced, selection = model.apply(molecule)
    assert selection.name is kind
    assert selection.electrons == electrons
    assert selection.orbitals == orbitals
    assert sum(reduced.num_particles) == electrons
    assert reduced.num_spatial_orbitals == len(orbitals)
    if kind is ModelKind.FULL:
        assert reduced is molecule


@pytest.mark.parametrize(
    ("model", "message"),
    [
        (ActiveSpace(3, (1, 2, 5)), "even electron count"),
        (ActiveSpace(6, (0, 1, 2, 3)), "molecule's electron count"),
        (ActiveSpace(2, (1, 2, 6)), "orbital index"),
        (ActiveSpace(2, ()), "orbital index"),
        (ActiveSpace(2, (1,)), "unoccupied spin orbitals"),
    ],
)
def test_rejects_invalid_model(molecule, model, message):
    with pytest.raises(ValueError, match=message):
        model.apply(molecule)


@pytest.mark.parametrize(
    ("filename", "electrons", "orbitals"),
    [
        ("lih.toml", 2, 4),
        ("lih-full.toml", 4, 6),
        ("lih-frozen-core.toml", 2, 5),
        ("lih-four-qubit.toml", 2, 3),
        ("lih-six-qubit.toml", 2, 4),
    ],
)
def test_example_configurations(molecule, filename, electrons, orbitals):
    config = load_config(f"configs/{filename}")
    selection = make_model(config.model).select(molecule)
    assert selection.electrons == electrons
    assert len(selection.orbitals) == orbitals

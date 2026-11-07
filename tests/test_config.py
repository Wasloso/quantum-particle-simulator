import json
from dataclasses import asdict, replace

import pytest

from lih_vqe.config import ExperimentConfig, load_config
from lih_vqe.enums import AnsatzKind, MappingKind, ModelKind, Optimizer


@pytest.mark.parametrize("distance", [0, -1, float("nan"), float("inf")])
def test_rejects_invalid_distance(distance):
    config = ExperimentConfig()
    with pytest.raises(ValueError, match="distance"):
        replace(config, molecule=replace(config.molecule, distance=distance)).validate()


def test_rejects_unsupported_reduction():
    config = ExperimentConfig()
    with pytest.raises(ValueError, match="parity"):
        replace(config, model=replace(config.model, mapping="jordan_wigner")).validate()


def test_misspelled_settings_do_not_silently_use_defaults(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text("[vqe]\nmax_evaulations = 10\n")
    with pytest.raises(ValueError, match="max_evaulations"):
        load_config(path)


def test_toml_orbitals_and_defaults(tmp_path):
    path = tmp_path / "settings.toml"
    path.write_text('[model]\nactive_orbitals = [1, 2, 3, 5]\n[vqe]\nansatz = "hea"\n')
    config = load_config(path)
    assert config.model.active_orbitals == (1, 2, 3, 5)
    assert config.model.kind is ModelKind.ACTIVE
    assert config.model.mapping is MappingKind.PARITY
    assert config.vqe.ansatz is AnsatzKind.HEA
    assert config.vqe.optimizer is Optimizer.SLSQP
    assert config.molecule.distance == 1.6
    saved = json.loads(json.dumps(asdict(config)))
    assert saved["model"]["kind"] == "active"
    assert saved["model"]["mapping"] == "parity"
    assert saved["vqe"]["ansatz"] == "hea"
    assert saved["vqe"]["optimizer"] == "SLSQP"

import pytest

from lih_vqe.experiments import run_experiment


@pytest.mark.integration
def test_run_records_energy_errors_and_sector(hydrogen_config):
    run = run_experiment(hydrogen_config)
    assert run["status"] == "completed"
    assert run["sector_valid"]
    assert abs(run["errors_millihartree"]["active"]) < 0.001
    assert run["hamiltonian"]["pauli_terms"]
    assert run["resources"]["parameters"] == len(run["vqe"]["parameters"])

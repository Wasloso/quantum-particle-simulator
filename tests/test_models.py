import pytest

from lih_vqe.config import ExperimentConfig
from lih_vqe.experiments import compare_models


@pytest.mark.integration
def test_reduction_ladder_and_exact_symmetry_reductions():
    rows = list(compare_models(ExperimentConfig()))
    assert [row["model"]["num_qubits"] for row in rows] == [12, 10, 10, 8, 6, 4]
    assert rows[0]["model"]["active_particles"] == [2, 2]
    assert rows[2]["model"]["active_particles"] == [1, 1]
    assert rows[0]["model_error_millihartree"] == pytest.approx(0, abs=1e-6)
    for before, after in [(rows[0], rows[1]), (rows[2], rows[3])]:
        assert before["references_hartree"]["mapped_sector"] == pytest.approx(
            after["references_hartree"]["mapped_sector"], abs=1e-8
        )
    assert rows[4]["model_error_millihartree"] < rows[5]["model_error_millihartree"]

from dataclasses import replace

import pytest

from lih_vqe.chemistry import build_problem


@pytest.mark.integration
def test_lih_mapping_agrees_with_independent_casci(lih_problem):
    refs = lih_problem.references
    assert lih_problem.hamiltonian.num_qubits == 4
    assert refs["mapped_sector"] == pytest.approx(refs["active_casci"], abs=1e-8)
    assert refs["full_fci"] < refs["active_casci"] < refs["hartree_fock"]
    assert 0 < refs["active_casci"] - refs["full_fci"] < 0.002


@pytest.mark.integration
def test_jordan_wigner_and_reduced_parity_match(hydrogen_config, hydrogen_problem):
    jw_config = replace(
        hydrogen_config,
        model=replace(hydrogen_config.model, mapping="jordan_wigner", two_qubit_reduction=False),
    )
    jw = build_problem(jw_config)
    assert jw.hamiltonian.num_qubits == 4
    assert hydrogen_problem.hamiltonian.num_qubits == 2
    assert jw.references["mapped_sector"] == pytest.approx(
        hydrogen_problem.references["mapped_sector"], abs=1e-8
    )

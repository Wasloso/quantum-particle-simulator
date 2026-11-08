from dataclasses import replace

import pytest

import lih_vqe.experiments as experiments
from lih_vqe.config import ScanConfig
from lih_vqe.experiments import scan


@pytest.mark.integration
def test_scan_reports_failed_geometry_and_continues(hydrogen_config, monkeypatch):
    config = replace(hydrogen_config, scan=ScanConfig(distances=(1.0, 0.735)))
    original = experiments.run_experiment

    def fail_first(current):
        if current.molecule.distance == 1.0:
            raise RuntimeError("simulated solver failure")
        return original(current)

    monkeypatch.setattr(experiments, "run_experiment", fail_first)
    runs = list(scan(config))
    assert runs[0]["status"] == "failed"
    assert runs[0]["error"] == "simulated solver failure"
    assert runs[1]["status"] == "completed"

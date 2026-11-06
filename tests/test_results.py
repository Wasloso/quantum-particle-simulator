import json

import pytest

from lih_vqe.enums import RunStatus
from lih_vqe.experiments import new_results, save_results


def test_result_schema_round_trip(tmp_path):
    results = new_results("run")
    assert results["status"] is RunStatus.RUNNING
    assert results["software"]["qiskit"]
    assert results["units"]["energy"] == "hartree"
    path = tmp_path / "nested" / "result.json"
    save_results(path, results)
    assert json.loads(path.read_text())["status"] == "running"
    assert not path.with_suffix(".json.tmp").exists()


def test_rejects_nonfinite_json_without_replacing_previous_result(tmp_path):
    path = tmp_path / "result.json"
    save_results(path, {"energy": 1.0})
    with pytest.raises(ValueError):
        save_results(path, {"energy": float("nan")})
    assert json.loads(path.read_text()) == {"energy": 1.0}

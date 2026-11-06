import json
import platform
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from lih_vqe.enums import RunStatus


def new_results(kind: str) -> dict:
    return {
        "schema_version": 1,
        "kind": kind,
        "status": RunStatus.RUNNING,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "units": {"distance": "angstrom", "energy": "hartree", "error": "millihartree"},
        "software": {
            "python": platform.python_version(),
            **{
                name: version(name)
                for name in [
                    "lih-vqe",
                    "numpy",
                    "qiskit",
                    "qiskit-nature",
                    "qiskit-aer",
                    "pyscf",
                    "scipy",
                ]
            },
        },
        "runs": [],
    }


def save_results(path: str | Path, results: dict) -> None:
    path = Path(path)
    serialized = json.dumps(results, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(path)

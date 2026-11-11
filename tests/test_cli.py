import pytest

from lih_vqe.cli import main


def test_existing_output_is_not_overwritten(tmp_path, capsys):
    path = tmp_path / "existing.json"
    path.write_text("keep me")
    with pytest.raises(SystemExit) as error:
        main(["run", "--output", str(path)])
    assert error.value.code == 1
    assert path.read_text() == "keep me"
    assert "already exists" in capsys.readouterr().err

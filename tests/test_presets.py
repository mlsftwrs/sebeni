"""Issue 10: presets ship inside the package; an unknown preset fails loudly."""

from typer.testing import CliRunner

from beni.cli.main import PRESETS_DIR, _preset_yaml, app, available_presets


def test_presets_resolve_inside_the_package():
    for name in ("single", "mali13", "multi13", "all"):
        path = _preset_yaml(name)
        assert path.is_file() and path.parent == PRESETS_DIR
    assert PRESETS_DIR.parts[-3:] == ("beni", "data", "presets")
    assert {"mali13", "single"} <= set(available_presets())


def test_unknown_preset_exits_2(tmp_path):
    result = CliRunner().invoke(app, ["exp", "--preset", "nope", "-w", str(tmp_path)])
    assert result.exit_code == 2
    assert "Unknown preset 'nope'" in result.output and "single" in result.output


def test_package_data_lists_presets():
    from pathlib import Path

    try:
        import tomllib
    except ImportError:  # Python 3.10
        import tomli as tomllib  # type: ignore

    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert "data/presets/*.yaml" in data["tool"]["setuptools"]["package-data"]["beni"]

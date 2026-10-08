import yaml

from scripts.run_matrix import build_config


def test_build_config_applies_requested_quantization(tmp_path):
    base_config_path = tmp_path / "base.yaml"
    base_config_path.write_text(
        yaml.safe_dump({"model": {"load_in_4bit": True}}),
        encoding="utf-8",
    )
    model_entry = {"id": "example/model", "slug": "example"}
    common = {
        "base_config_path": base_config_path,
        "model_entry": model_entry,
        "algorithm": "grpo",
        "scope": "MULTI13",
        "max_steps": 1,
        "distil_model": "teacher",
        "location": "global",
        "vertex": False,
        "backend": "algorithmic",
        "run_dir": tmp_path / "run",
    }

    four_bit = build_config(**common, quantization="4bit")["model"]
    eight_bit = build_config(**common, quantization="8bit")["model"]
    unquantized = build_config(**common, quantization="none")["model"]

    assert four_bit["load_in_4bit"] is True
    assert four_bit["load_in_8bit"] is False
    assert eight_bit["load_in_4bit"] is False
    assert eight_bit["load_in_8bit"] is True
    assert unquantized["load_in_4bit"] is False
    assert unquantized["load_in_8bit"] is False


def test_build_config_without_quantization_preserves_base_config(tmp_path):
    base_config_path = tmp_path / "base.yaml"
    base_config_path.write_text(
        yaml.safe_dump({"model": {"load_in_4bit": True}}),
        encoding="utf-8",
    )

    cfg = build_config(
        base_config_path=base_config_path,
        model_entry={"id": "example/model", "slug": "example"},
        algorithm="grpo",
        scope="MULTI13",
        max_steps=1,
        distil_model="teacher",
        location="global",
        vertex=False,
        backend="algorithmic",
        run_dir=tmp_path / "run",
    )

    assert cfg["model"]["load_in_4bit"] is True


def test_build_config_rejects_unknown_quantization(tmp_path):
    base_config_path = tmp_path / "base.yaml"
    base_config_path.write_text("{}", encoding="utf-8")

    try:
        build_config(
            base_config_path=base_config_path,
            model_entry={"id": "example/model", "slug": "example"},
            algorithm="grpo",
            scope="MULTI13",
            max_steps=1,
            distil_model="teacher",
            location="global",
            vertex=False,
            backend="algorithmic",
            run_dir=tmp_path / "run",
            quantization="2bit",
        )
    except ValueError as exc:
        assert "Unsupported quantization mode" in str(exc)
    else:
        raise AssertionError("Unsupported quantization mode should be rejected")

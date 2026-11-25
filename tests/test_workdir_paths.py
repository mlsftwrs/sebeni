"""Issue 23: default output_dir / runtime_dir follow the resolved working dir."""

from beni.core.srl.config import MasterConfig
from beni.utils import config as cfg


def test_defaults_follow_new_workdir_after_another_was_active(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    cfg.set_working_dir(first, ensure=True)
    mc = MasterConfig()
    mc.working_dir = str(second)
    mc.apply_workdir()
    for block in (mc.trainer, mc.sft, mc.dpo, mc.apo):
        assert block.output_dir == str(second / "models")
    assert mc.processor.runtime_dir == str(second / "runtime")


def test_yaml_output_dir_is_kept(tmp_path):
    cfg.set_working_dir(tmp_path / "first", ensure=True)
    config = tmp_path / "c.yaml"
    custom = tmp_path / "elsewhere"
    config.write_text(
        f"working_dir: {tmp_path / 'second'}\nsft:\n  output_dir: {custom}\n", encoding="utf-8"
    )
    mc = MasterConfig.from_yaml(config)
    assert mc.sft.output_dir == str(custom)
    assert mc.trainer.output_dir == str(tmp_path / "second" / "models")


def test_two_configs_in_one_process(tmp_path):
    a, b = tmp_path / "a.yaml", tmp_path / "b.yaml"
    a.write_text(f"working_dir: {tmp_path / 'run_a'}\n", encoding="utf-8")
    b.write_text(f"working_dir: {tmp_path / 'run_b'}\n", encoding="utf-8")
    MasterConfig.from_yaml(a)
    mc_b = MasterConfig.from_yaml(b)
    assert mc_b.trainer.output_dir == str(tmp_path / "run_b" / "models")


def test_cli_workdir_override_moves_defaults(tmp_path):
    config = tmp_path / "c.yaml"
    config.write_text(f"working_dir: {tmp_path / 'yaml_wd'}\n", encoding="utf-8")
    mc = MasterConfig.from_yaml(config, working_dir=tmp_path / "cli_wd")
    assert mc.sft.output_dir == str(tmp_path / "cli_wd" / "models")

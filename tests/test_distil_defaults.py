"""Issue 13: one default distillation model for every entry point."""

import inspect

import yaml

from beni.cli.main import default_config_yaml
from beni.core.morphotactic.distil.distillation import Distiller
from beni.core.morphotactic.distil.providers.google import GoogleProvider
from beni.core.srl.config import DistillationConfig
from beni.utils import config as cfg


def test_all_defaults_are_the_constant():
    expected = cfg.DEFAULT_DISTIL_MODEL
    assert DistillationConfig().model == expected
    assert inspect.signature(Distiller.__init__).parameters["model"].default == expected
    assert inspect.signature(GoogleProvider.__init__).parameters["model"].default == expected
    scaffold = yaml.safe_load(default_config_yaml(["bam"], "/tmp/x"))
    assert scaffold["distillation"]["model"] == expected


def test_shipped_presets_match():
    for path in ("configs/exp.yaml", "beni/data/exp.yaml", "beni/data/presets/multi13.yaml", "beni/data/presets/single.yaml"):
        data = yaml.safe_load(open(path, encoding="utf-8"))
        assert data["distillation"]["model"] == cfg.DEFAULT_DISTIL_MODEL, path

"""Regression tests for config loading.

Guards the PyYAML 6+ breakage that previously crashed ``test.py`` /
``train.py`` with ``TypeError: load() missing 1 required positional
argument: 'Loader'``. Every shipped config must parse with the same
``yaml.safe_load`` call the scripts now use, and the flow config must
declare the optical-flow keys FlowCLIP relies on.
"""
import glob
import os

import pytest

yaml = pytest.importorskip("yaml")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_FILES = sorted(
    glob.glob(os.path.join(REPO_ROOT, "configs", "**", "*.yaml"), recursive=True)
)


def test_configs_are_present():
    assert CONFIG_FILES, "no config files found under configs/"


@pytest.mark.parametrize("path", CONFIG_FILES, ids=lambda p: os.path.relpath(p, REPO_ROOT))
def test_config_parses_with_safe_load(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    assert isinstance(cfg, dict)
    assert "network" in cfg and "data" in cfg


def test_flow_config_declares_flow_keys():
    path = os.path.join(REPO_ROOT, "configs", "hmdb51", "hmdb_flow.yaml")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    data = cfg["data"]
    assert data.get("use_flow") is True
    assert "flow_tmpl" in data
    assert "flow_root" in data
    assert "Flow" in str(data.get("modality", "")), "modality should include Flow"

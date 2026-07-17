"""Contracts for the standalone AudioDec training configurations."""

from __future__ import annotations

import copy
import importlib.util
import subprocess
from pathlib import Path

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "config" / "autoencoder"
BASELINE_PATH = CONFIG_ROOT / "symAD_libritts_24000_hop300.yaml"
CONFIG_MATRIX = {
    "symAD_libritts_24000_hop300_train-clean-460_9600.yaml": (
        "train-clean-460",
        9600,
        0,
    ),
    "symAD_libritts_24000_hop300_train-clean-460_9600_context15900.yaml": (
        "train-clean-460",
        9600,
        15900,
    ),
    "symAD_libritts_24000_hop300_train-clean-460_normalized_9600.yaml": (
        "train-clean-460_normalized",
        9600,
        0,
    ),
    "symAD_libritts_24000_hop300_train-clean-460_normalized_9600_context15900.yaml": (
        "train-clean-460_normalized",
        9600,
        15900,
    ),
}


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_audiodec_utils():
    module_path = PROJECT_ROOT / "bin" / "utils.py"
    spec = importlib.util.spec_from_file_location("audiodec_bin_utils", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_resolved_config(path: Path) -> dict:
    return _load_audiodec_utils().load_training_config(path)


@pytest.mark.parametrize(("filename", "expected"), CONFIG_MATRIX.items())
def test_training_config_matrix_matches_baseline(
    filename: str, expected: tuple[str, int, int]
) -> None:
    """Each variant changes only its tag, subset, and context contract."""
    expected_subset, expected_length, expected_context = expected
    baseline = _load_yaml(BASELINE_PATH)
    variant_path = CONFIG_ROOT / filename
    raw_config = _load_yaml(variant_path)
    config = _load_resolved_config(variant_path)

    assert raw_config["base_config"] == BASELINE_PATH.name
    assert len(raw_config) <= 6

    assert config["data"] == {
        "path": "/workspace/data/libritts",
        "subset": {
            "train": expected_subset,
            "valid": "test-clean",
            "test": "test-clean",
        },
    }
    assert config["batch_length"] == expected_length
    assert config["adv_batch_length"] == expected_length
    assert config.get("context_length", 0) == expected_context
    assert config["batch_size"] == 16
    assert expected_length % 300 == 0
    assert expected_context % 300 == 0
    assert config["tag"] == f"autoencoder/{Path(filename).stem}"

    comparable = copy.deepcopy(config)
    comparable.pop("tag")
    comparable["data"]["subset"]["train"] = "train-clean-460"
    comparable["batch_length"] = 9600
    comparable["adv_batch_length"] = 9600
    comparable.pop("context_length", None)
    assert comparable == baseline


def test_training_configs_have_distinct_checkpoint_directories() -> None:
    """Every matrix member resolves beneath the experiment root uniquely."""
    tags = {
        _load_resolved_config(CONFIG_ROOT / filename)["tag"]
        for filename in CONFIG_MATRIX
    }

    assert len(tags) == len(CONFIG_MATRIX)
    assert len({Path("/workspace/data/audiodec/exp") / tag for tag in tags}) == 4


def test_experiment_tag_resolution() -> None:
    """CLI tags override YAML tags and absence is an explicit error."""
    utils = _load_audiodec_utils()

    assert utils.resolve_experiment_tag({"tag": "from-config"}) == "from-config"
    assert (
        utils.resolve_experiment_tag({"tag": "from-config"}, "from-cli")
        == "from-cli"
    )
    with pytest.raises(ValueError, match="Pass --tag or define 'tag'"):
        utils.resolve_experiment_tag({})


def test_training_config_rejects_circular_inheritance(tmp_path: Path) -> None:
    """Circular baseline references fail explicitly instead of recursing forever."""
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("base_config: second.yaml\n", encoding="utf-8")
    second.write_text("base_config: first.yaml\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Circular base_config inheritance"):
        _load_audiodec_utils().load_training_config(first)


def test_docker_training_script_routes_explicit_configs() -> None:
    """The wrapper preserves defaults only when no config was supplied."""
    script_path = PROJECT_ROOT / "scripts" / "docker" / "train_libritts.sh"
    script = script_path.read_text(encoding="utf-8")

    subprocess.run(["bash", "-n", str(script_path)], check=True)
    assert '-c|--config|--config=*) explicit_config=true' in script
    assert 'if [[ "${explicit_config}" == false ]]' in script
    assert '"${training_args[@]}"' in script

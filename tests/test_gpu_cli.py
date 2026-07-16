"""Contracts for selecting an AudioDec training GPU from the CLI."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# The nested project is intentionally loaded from its own repository root.
# pylint: disable=import-error,wrong-import-position
from bin.train import add_gpu_argument, resolve_training_device


def _gpu_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_gpu_argument(parser)
    return parser


@pytest.mark.parametrize("gpu_index", [0, 3])
def test_gpu_argument_accepts_supported_boundary_indices(gpu_index: int) -> None:
    """The lowest and highest supported GPU indices are accepted."""
    assert _gpu_parser().parse_args(["--gpu", str(gpu_index)]).gpu == gpu_index


@pytest.mark.parametrize("arguments", [[], ["--gpu", "-1"], ["--gpu", "4"]])
def test_gpu_argument_rejects_missing_or_unsupported_indices(
    arguments: list[str],
) -> None:
    """A missing GPU or an index outside 0-3 fails argument parsing."""
    with pytest.raises(SystemExit):
        _gpu_parser().parse_args(arguments)


def test_requested_gpu_controls_cuda_device_and_log(monkeypatch, caplog) -> None:
    """The requested index determines the CUDA device and status log."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    with caplog.at_level(logging.INFO):
        device = resolve_training_device(2)

    assert device == torch.device("cuda:2")
    assert "device: gpu 2" in caplog.messages


def test_device_resolution_preserves_cpu_fallback(monkeypatch, caplog) -> None:
    """Training continues to select CPU on systems without CUDA."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with caplog.at_level(logging.INFO):
        device = resolve_training_device(3)

    assert device == torch.device("cpu")
    assert "device: cpu" in caplog.messages

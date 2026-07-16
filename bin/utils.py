#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""Utility modules."""

import os
import yaml


def resolve_experiment_tag(config, cli_tag=None):
    """Resolve the experiment tag, preferring an explicit CLI override."""
    tag = cli_tag if cli_tag is not None else config.get("tag")
    if not tag:
        raise ValueError(
            "No experiment tag was provided. Pass --tag or define 'tag' in the config."
        )
    return tag


def _deep_merge(base, overrides):
    """Merge nested config mappings without modifying either input."""
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_training_config(config_path, config_chain=()):
    """Load a training config and recursively merge its relative baseline."""
    config_path = os.path.realpath(config_path)
    if config_path in config_chain:
        chain = " -> ".join(config_chain + (config_path,))
        raise ValueError(f"Circular base_config inheritance: {chain}")

    with open(config_path, encoding="utf-8") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    if not isinstance(config, dict):
        raise ValueError(f"Training config must contain a mapping: {config_path}")

    base_config = config.pop("base_config", None)
    if base_config is None:
        return config
    if not isinstance(base_config, str) or not base_config:
        raise ValueError(f"base_config must be a non-empty path: {config_path}")

    if not os.path.isabs(base_config):
        base_config = os.path.join(os.path.dirname(config_path), base_config)
    base = load_training_config(base_config, config_chain + (config_path,))
    return _deep_merge(base, config)


def load_config(checkpoint, config_name='config.yml'):
    dirname = os.path.dirname(checkpoint)
    config_path = os.path.join(dirname, config_name)
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    return config

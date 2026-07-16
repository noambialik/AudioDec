"""Checkpoint writing helpers for AudioDec training."""

import logging
import time

import torch


def retried_save_checkpoint(
    state_dict,
    checkpoint_path,
    retry_attempts=20,
    retry_sleep_seconds=60.0,
):
    """Save a checkpoint, retrying transient filesystem and PyTorch failures."""
    total_attempts = retry_attempts + 1

    for attempt in range(1, total_attempts + 1):
        try:
            torch.save(state_dict, checkpoint_path)
        except (OSError, RuntimeError):
            logging.exception(
                "Checkpoint save failed (attempt %d/%d): %s",
                attempt,
                total_attempts,
                checkpoint_path,
            )
            if attempt == total_attempts:
                raise
            time.sleep(retry_sleep_seconds)
        else:
            return

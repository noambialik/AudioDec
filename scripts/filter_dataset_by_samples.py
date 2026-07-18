#!/usr/bin/env python3
"""Replace a dataset with links to audio files of sufficient length."""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

import soundfile as sound_file


DEFAULT_MIN_SAMPLES = 25_500
AUDIO_EXTENSIONS = {".wav", ".flac"}


def qualifying_audio_files(
    dataset: Path, min_samples: int
) -> tuple[list[Path], int]:
    """Return relative paths meeting the threshold and the short-file count."""
    qualifying: list[Path] = []
    skipped = 0
    audio_files = sorted(
        path
        for path in dataset.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )

    for path in audio_files:
        frames = int(sound_file.info(path).frames)
        if frames >= min_samples:
            qualifying.append(path.relative_to(dataset))
        else:
            skipped += 1

    return qualifying, skipped


def filter_dataset(dataset: Path, min_samples: int) -> None:
    """Move the dataset aside and publish a filtered symbolic-link tree."""
    dataset = dataset.absolute()

    if min_samples <= 0:
        raise ValueError("--min-samples must be positive")
    if dataset.is_symlink():
        raise ValueError(f"Dataset path must not be a symbolic link: {dataset}")
    if not dataset.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset}")

    backup = dataset.with_name(f"_{dataset.name}")
    if backup.exists() or backup.is_symlink():
        raise FileExistsError(f"Backup path already exists: {backup}")

    qualifying, skipped = qualifying_audio_files(dataset, min_samples)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{dataset.name}-filtered-", dir=dataset.parent)
    )

    try:
        for relative_path in qualifying:
            staged_link = staging / relative_path
            final_link = dataset / relative_path
            target = backup / relative_path

            staged_link.parent.mkdir(parents=True, exist_ok=True)
            relative_target = os.path.relpath(target, start=final_link.parent)
            staged_link.symlink_to(relative_target)

        dataset.rename(backup)
        try:
            staging.rename(dataset)
        except Exception:
            backup.rename(dataset)
            raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    print(f"Original dataset moved to: {backup}")
    print(f"Filtered link tree created at: {dataset}")
    print(f"Linked files: {len(qualifying)}")
    print(f"Skipped short files: {skipped}")


def main() -> None:
    """Parse command-line arguments and filter one dataset."""
    parser = argparse.ArgumentParser(
        description=(
            "Move a dataset to an underscore-prefixed path and replace it with "
            "symbolic links to sufficiently long WAV/FLAC files."
        )
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument(
        "--min-samples",
        type=int,
        default=DEFAULT_MIN_SAMPLES,
    )
    args = parser.parse_args()
    filter_dataset(args.dataset, args.min_samples)


if __name__ == "__main__":
    main()

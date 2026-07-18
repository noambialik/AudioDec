#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Reference (https://github.com/kan-bayashi/ParallelWaveGAN/)

"""Customized collater modules for Pytorch DataLoader."""

import torch
import numpy as np


class CollaterAudio(object):
    """Customized collater for loading single audio."""

    def __init__(
        self,
        batch_length=9600,
        context_length=0,
        sampling_context_length=None,
        random_segment=True,
    ):
        """
        Args:
            batch_length (int): The length of audio signal batch.
            context_length (int): Length of real audio prepended for warm-up.
            sampling_context_length (int): Prefix sampled before the target.
            random_segment (bool): Randomly choose the segment start when true.

        """
        if sampling_context_length is None:
            sampling_context_length = context_length
        if context_length < 0:
            raise ValueError("context_length must be non-negative")
        if sampling_context_length < context_length:
            raise ValueError(
                "sampling_context_length must be at least context_length"
            )
        self.sampling_context_length = sampling_context_length
        self.batch_length = batch_length
        self.sample_length = sampling_context_length + batch_length
        self.trim_length = sampling_context_length - context_length
        self.random_segment = random_segment
        if sampling_context_length == 0:
            # Preserve the legacy strict `len(audio) > batch_length` contract.
            self.minimum_frames = self.sample_length + 1
        else:
            self.minimum_frames = self.sample_length


    def __call__(self, batch):
        for index, audio in enumerate(batch):
            self._validate_audio_length(audio, index)
        
        # random cut
        starts, ends = self._random_segment(batch)
        x_batch = self._cut(batch, starts + self.trim_length, ends)
        
        return x_batch


    def _validate_audio_length(self, audio, batch_index):
        if len(audio) < self.minimum_frames:
            raise ValueError(
                f"Audio at batch index {batch_index} has {len(audio)} samples; "
                f"at least {self.minimum_frames} are required"
            )
    

    def _random_segment(self, xs):
        if not self.random_segment:
            starts = np.zeros(len(xs), dtype=int)
            return starts, starts + self.sample_length

        start_offsets = []
        for x in xs:
            last_valid_start = len(x) - self.sample_length
            if self.sampling_context_length == 0:
                # Preserve the original zero-context sampling behavior.
                start_upper_bound = last_valid_start
            else:
                # Include the final valid start position in context mode.
                start_upper_bound = last_valid_start + 1
            start_offsets.append(np.random.randint(0, start_upper_bound))

        starts = np.array(start_offsets)
        ends = starts + self.sample_length
        return starts, ends
    

    def _cut(self, xs, starts, ends):
        x_batch = np.array([x[start:end] for x, start, end in zip(xs, starts, ends)])
        x_batch = torch.tensor(x_batch, dtype=torch.float).transpose(2, 1)  # (B, C, T)
        return x_batch


class CollaterAudioPair(CollaterAudio):
    """Customized collater for loading audio pair."""

    def __call__(self, batch):
        for index, pair in enumerate(batch):
            if len(pair[0]) != len(pair[1]):
                raise ValueError(
                    f"Audio pair at batch index {index} is misaligned: "
                    f"{len(pair[0])} != {len(pair[1])} samples"
                )
            self._validate_audio_length(pair[0], index)
        xs, ns = [b[0] for b in batch], [b[1] for b in batch]

        # random cut
        starts, ends = self._random_segment(xs)
        starts = starts + self.trim_length
        x_batch = self._cut(xs, starts, ends)
        n_batch = self._cut(ns, starts, ends)
        
        return n_batch, x_batch # (input, output)

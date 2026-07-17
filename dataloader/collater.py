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
    ):
        """
        Args:
            batch_length (int): The length of audio signal batch.
            context_length (int): Length of real audio prepended for warm-up.

        """
        if context_length < 0:
            raise ValueError("context_length must be non-negative")
        self.batch_length = batch_length
        self.context_length = context_length
        self.segment_length = context_length + batch_length


    def __call__(self, batch):
        # filter short batch
        if self.context_length == 0:
            xs = [b for b in batch if len(b) > self.segment_length]
        else:
            xs = [b for b in batch if len(b) >= self.segment_length]
        if not xs:
            raise ValueError(
                f"No audio is at least {self.segment_length} samples long"
            )
        
        # random cut
        starts, ends = self._random_segment(xs)
        x_batch = self._cut(xs, starts, ends)
        
        return x_batch
    

    def _random_segment(self, xs):
        start_offsets = []
        for x in xs:
            last_valid_start = len(x) - self.segment_length
            if self.context_length == 0:
                # Preserve the original zero-context sampling behavior.
                start_upper_bound = last_valid_start
            else:
                # Include the final valid start position in context mode.
                start_upper_bound = last_valid_start + 1
            start_offsets.append(np.random.randint(0, start_upper_bound))

        starts = np.array(start_offsets)
        ends = starts + self.segment_length
        return starts, ends
    

    def _cut(self, xs, starts, ends):
        x_batch = np.array([x[start:end] for x, start, end in zip(xs, starts, ends)])
        x_batch = torch.tensor(x_batch, dtype=torch.float).transpose(2, 1)  # (B, C, T)
        return x_batch


class CollaterAudioPair(CollaterAudio):
    """Customized collater for loading audio pair."""

    def __init__(
        self,
        batch_length=9600,
        context_length=0,
    ):
        super().__init__(
            batch_length=batch_length,
            context_length=context_length,
        )


    def __call__(self, batch):
        if self.context_length == 0:
            batch = [
                b for b in batch
                if (len(b[0]) > self.segment_length) and (len(b[0]) == len(b[1]))
            ]
        else:
            batch = [
                b for b in batch
                if (len(b[0]) >= self.segment_length) and (len(b[0]) == len(b[1]))
            ]
        if not batch:
            raise ValueError(
                f"No aligned audio pair is at least {self.segment_length} samples long"
            )
        xs, ns = [b[0] for b in batch], [b[1] for b in batch]

        # random cut
        starts, ends = self._random_segment(xs)
        x_batch = self._cut(xs, starts, ends)
        n_batch = self._cut(ns, starts, ends)
        
        return n_batch, x_batch # (input, output)

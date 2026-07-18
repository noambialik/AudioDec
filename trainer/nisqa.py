"""Scheduled, validation-only NISQA evaluation."""

import logging

from torchmetrics.audio import NonIntrusiveSpeechQualityAssessment


class NISQAValidationEvaluator:
    """Own NISQA scheduling, accumulation, validation, and aggregation."""

    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.evaluation_interval_steps = 100_000
        self.max_clips = 64
        self.mos_index = 0
        logging.info(
            "Initializing NISQA; TorchMetrics may download pretrained "
            "weights on first scheduled evaluation."
        )
        self._metric = NonIntrusiveSpeechQualityAssessment(fs=self.sample_rate)
        self._clip_count = 0
        self._active = False

    @property
    def clip_count(self):
        """Return the number of clips accumulated in the current evaluation."""
        return self._clip_count

    def start_evaluation(self, step):
        """Activate and reset NISQA only at fixed 100,000-step boundaries."""
        self._active = (
            step > 0 and step % self.evaluation_interval_steps == 0
        )
        if not self._active:
            return False

        self._clip_count = 0
        self._metric.reset()
        return True

    def update(self, predicted_audio):
        """Accumulate at most the first 64 mono validation clips."""
        if not self._active or self._clip_count >= self.max_clips:
            return
        if predicted_audio.ndim != 3 or predicted_audio.shape[1] != 1:
            raise ValueError(
                "NISQA expects mono validation audio shaped [batch, 1, samples]."
            )

        remaining = self.max_clips - self._clip_count
        selected_audio = predicted_audio[:remaining, 0].detach()
        self._metric.update(selected_audio)
        self._clip_count += selected_audio.shape[0]

    def compute_mos(self):
        """Return overall MOS for the active validation epoch."""
        if not self._active:
            raise RuntimeError("NISQA is not scheduled for this validation step.")
        if self._clip_count == 0:
            raise RuntimeError("NISQA received no validation clips to score.")

        mos = self._metric.compute()[self.mos_index].item()
        self._active = False
        return mos

"""Contracts for validation-only NISQA reporting."""

from collections import defaultdict
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

tensorboard_stub = ModuleType("tensorboardX")
tensorboard_stub.SummaryWriter = object
sys.modules.setdefault("tensorboardX", tensorboard_stub)

from codecTrain import TrainMain
from dataloader.collater import CollaterAudio
import trainer.nisqa as nisqa_module
from trainer.nisqa import NISQAValidationEvaluator
from trainer.trainerGAN import TrainerGAN


class StubTrainer(TrainerGAN):
    """Concrete test seam for shared trainer behavior."""

    def _train_step(self, batch):
        raise NotImplementedError

    def _eval_step(self, batch):
        raise NotImplementedError


class FakeNISQA:
    """Record metric lifecycle calls without loading model weights."""

    def __init__(self):
        self.updates = []
        self.reset_count = 0

    def update(self, audio):
        self.updates.append(audio.detach().clone())

    def compute(self):
        return torch.tensor([3.25, 2.0, 2.1, 2.2, 2.3])

    def reset(self):
        self.reset_count += 1


def _trainer_with_fake_nisqa():
    trainer = object.__new__(StubTrainer)
    trainer.config = {
        "sampling_rate": 24000,
        "use_mel_loss": False,
        "use_stft_loss": False,
        "use_shape_loss": False,
    }
    evaluator = NISQAValidationEvaluator(sample_rate=24000)
    evaluator._metric = FakeNISQA()
    evaluator.start_evaluation(evaluator.evaluation_interval_steps)
    trainer.nisqa_evaluator = evaluator
    return trainer, evaluator


def test_nisqa_is_validation_only_and_does_not_change_metric_loss():
    trainer, evaluator = _trainer_with_fake_nisqa()
    predicted = torch.zeros(8, 1, 24000)

    train_loss = trainer._metric_loss(predicted, predicted, mode="train")
    eval_loss = trainer._metric_loss(predicted, predicted, mode="eval")

    assert train_loss == eval_loss == 0.0
    assert len(evaluator._metric.updates) == 1
    assert evaluator.clip_count == 8


def test_nisqa_is_created_on_init_with_the_training_sample_rate(monkeypatch):
    created_sample_rates = []
    fake_metric = FakeNISQA()

    def create_metric(*, fs):
        created_sample_rates.append(fs)
        return fake_metric

    monkeypatch.setattr(
        nisqa_module,
        "NonIntrusiveSpeechQualityAssessment",
        create_metric,
    )

    evaluator = NISQAValidationEvaluator(sample_rate=24000)

    assert evaluator._metric is fake_metric
    assert created_sample_rates == [24000]


def test_nisqa_runs_only_at_100000_step_boundaries():
    evaluator = NISQAValidationEvaluator(sample_rate=24000)

    assert evaluator.evaluation_interval_steps == 100_000
    assert evaluator.max_clips == 64
    assert evaluator.mos_index == 0
    assert evaluator.start_evaluation(0) is False
    assert evaluator.start_evaluation(99_999) is False
    assert evaluator.start_evaluation(100_000) is True
    assert evaluator.start_evaluation(199_999) is False
    assert evaluator.start_evaluation(200_000) is True


def test_nisqa_scores_exactly_the_first_64_clips():
    _, evaluator = _trainer_with_fake_nisqa()

    evaluator.update(torch.zeros(40, 1, 24000))
    evaluator.update(torch.ones(40, 1, 24000))
    evaluator.update(torch.full((8, 1, 24000), 2.0))

    assert evaluator.clip_count == evaluator.max_clips
    assert [batch.shape[0] for batch in evaluator._metric.updates] == [40, 24]
    torch.testing.assert_close(
        evaluator._metric.updates[1], torch.ones(24, 24000)
    )


@pytest.mark.parametrize(
    ("step", "expected_log"),
    ((99_999, {}), (100_000, {"eval/nisqa_mos": 3.25})),
)
def test_trainer_logs_nisqa_only_on_scheduled_steps(step, expected_log):
    class EvalTrainer(StubTrainer):
        def _eval_step(self, batch):
            self._metric_loss(batch, batch, mode="eval")

    trainer = object.__new__(EvalTrainer)
    trainer.config = {
        "sampling_rate": 24000,
        "use_mel_loss": False,
        "use_stft_loss": False,
        "use_shape_loss": False,
    }
    trainer.steps = step
    trainer.model = {}
    trainer.data_loader = {"dev": [torch.zeros(1, 1, 24000)]}
    trainer.total_eval_loss = defaultdict(float)
    trainer.nisqa_evaluator = NISQAValidationEvaluator(sample_rate=24000)
    trainer.nisqa_evaluator._metric = FakeNISQA()
    logged = {}
    trainer._write_to_tensorboard = lambda values: logged.update(values)

    trainer._eval_epoch()

    assert logged == expected_log
    expected_reset_count = 1 if expected_log else 0
    expected_clip_count = 1 if expected_log else 0
    assert trainer.nisqa_evaluator._metric.reset_count == expected_reset_count
    assert trainer.nisqa_evaluator.clip_count == expected_clip_count


def test_nisqa_rejects_multichannel_validation_audio():
    _, evaluator = _trainer_with_fake_nisqa()

    with pytest.raises(ValueError, match="expects mono validation audio"):
        evaluator.update(torch.zeros(1, 2, 24000))


def test_training_and_validation_use_distinct_segment_contracts():
    train_main = object.__new__(TrainMain)
    train_main.train_mode = "autoencoder"
    train_main.batch_length = 9600
    train_main.config = {
        "sampling_rate": 24000,
        "context_length": 15900,
        "sampling_context_length": 15900,
    }
    train_main._audio = lambda subset: [np.zeros((40000, 1))]
    captured = {}

    def capture_data_loader(dataset, train_collater, valid_collater):
        captured["train"] = train_collater
        captured["valid"] = valid_collater

    train_main._data_loader = capture_data_loader

    train_main.initialize_data_loader()

    assert captured["train"].batch_length == 9600
    assert captured["train"].random_segment is True
    assert captured["valid"].batch_length == 24000
    assert captured["valid"].random_segment is False


def test_validation_segment_selection_is_deterministic():
    audio = np.arange(30000, dtype=np.float32)[:, None]
    valid_collater = CollaterAudio(batch_length=24000, random_segment=False)

    first = valid_collater([audio])
    second = valid_collater([audio])

    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first[0, 0], torch.arange(24000, dtype=torch.float))

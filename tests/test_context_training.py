"""Contracts for steady-state context training."""

import copy
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

tensorboard_stub = ModuleType("tensorboardX")
tensorboard_stub.SummaryWriter = object
sys.modules.setdefault("tensorboardX", tensorboard_stub)

from dataloader.collater import CollaterAudio, CollaterAudioPair
from models.autoencoder.modules.quantizer import Quantizer
from trainer.autoencoder import Trainer


def _backward_conv(interval, kernel, stride=1, dilation=1):
    start, end = interval
    return (
        start * stride - (kernel - 1) * dilation,
        end * stride,
    )


def _backward_residual_stack(interval):
    start, end = interval
    return start - 6 * (1 + 3 + 9), end


def _backward_deconv(interval, stride):
    start, end = interval
    return start // stride - 1, end // stride


def _decoder_latent_interval(output_phase):
    interval = _backward_conv((output_phase, output_phase), kernel=7)
    for stride in (3, 4, 5, 5):
        interval = _backward_residual_stack(interval)
        interval = _backward_deconv(interval, stride)
    return _backward_conv(interval, kernel=7)


def test_context_covers_maximum_end_to_end_receptive_field() -> None:
    latent_spans = []
    end_to_end_spans = []
    for phase in range(300):
        start, end = _decoder_latent_interval(phase)
        latent_span = end - start + 1
        latent_spans.append(latent_span)
        end_to_end_spans.append(7209 + (latent_span - 1) * 300)

    assert min(latent_spans) == 28
    assert max(latent_spans) == 29
    assert min(end_to_end_spans) == 15309
    assert max(end_to_end_spans) == 15609
    assert 15900 >= max(end_to_end_spans)
    assert 15900 % 300 == 0


def test_audio_collater_prepends_context_to_supervised_batch() -> None:
    audio = np.arange(7, dtype=np.float32)[:, None]
    collater = CollaterAudio(batch_length=4, context_length=3)

    batch = collater([audio])

    assert batch.shape == (1, 1, 7)
    torch.testing.assert_close(batch[0, 0], torch.arange(7, dtype=torch.float))


def test_context_and_matched_control_use_the_same_target_window() -> None:
    audio = np.arange(10, dtype=np.float32)[:, None]
    np.random.seed(17)
    context_batch = CollaterAudio(
        batch_length=4,
        context_length=3,
        sampling_context_length=3,
    )([audio])
    np.random.seed(17)
    control_batch = CollaterAudio(
        batch_length=4,
        context_length=0,
        sampling_context_length=3,
    )([audio])

    assert context_batch.shape == (1, 1, 7)
    assert control_batch.shape == (1, 1, 4)
    torch.testing.assert_close(control_batch, context_batch[..., 3:])


def test_collater_rejects_model_context_larger_than_sampling_context() -> None:
    with pytest.raises(
        ValueError,
        match="sampling_context_length must be at least context_length",
    ):
        CollaterAudio(
            batch_length=4,
            context_length=3,
            sampling_context_length=2,
        )


def test_audio_pair_collater_uses_the_same_context_segment() -> None:
    clean = np.arange(7, dtype=np.float32)[:, None]
    noisy = clean + 100
    collater = CollaterAudioPair(batch_length=4, context_length=3)

    noisy_batch, clean_batch = collater([(clean, noisy)])

    assert noisy_batch.shape == clean_batch.shape == (1, 1, 7)
    torch.testing.assert_close(noisy_batch - clean_batch, torch.full_like(clean_batch, 100))


def test_collater_zero_context_keeps_the_supervised_length() -> None:
    audio = np.arange(5, dtype=np.float32)[:, None]

    batch = CollaterAudio(batch_length=4)([audio])

    assert batch.shape == (1, 1, 4)


def test_quantizer_zero_boundary_matches_the_original_call() -> None:
    original = Quantizer(code_dim=2, codebook_num=2, codebook_size=4)
    explicit_zero = copy.deepcopy(original)
    z = torch.randn(2, 2, 5)

    original_result = original(z)
    explicit_result = explicit_zero(z, num_context_frames=0)

    for original_value, explicit_value in zip(original_result, explicit_result):
        torch.testing.assert_close(original_value, explicit_value)
    for original_layer, explicit_layer in zip(
        original.codebook.layers,
        explicit_zero.codebook.layers,
    ):
        torch.testing.assert_close(
            original_layer.cluster_size,
            explicit_layer.cluster_size,
        )
        torch.testing.assert_close(original_layer.embed_avg, explicit_layer.embed_avg)


def test_collater_rejects_audio_shorter_than_context_and_batch() -> None:
    audio = np.arange(6, dtype=np.float32)[:, None]

    with pytest.raises(ValueError, match="at least 7"):
        CollaterAudio(batch_length=4, context_length=3)([audio])


def test_collater_does_not_silently_drop_a_short_clip() -> None:
    valid = np.arange(7, dtype=np.float32)[:, None]
    short = np.arange(6, dtype=np.float32)[:, None]

    with pytest.raises(ValueError, match="batch index 1 has 6 samples"):
        CollaterAudio(batch_length=4, context_length=3)([valid, short])


def test_audio_pair_collater_rejects_misaligned_clips() -> None:
    clean = np.arange(7, dtype=np.float32)[:, None]
    noisy = np.arange(8, dtype=np.float32)[:, None]

    with pytest.raises(ValueError, match="batch index 0 is misaligned: 7 != 8"):
        CollaterAudioPair(batch_length=4, context_length=3)([(clean, noisy)])


def test_required_audio_length_preserves_context_and_legacy_boundaries() -> None:
    assert CollaterAudio(9600, 15900).minimum_frames == 25500
    assert CollaterAudio(9600, 0).minimum_frames == 9601


def test_quantizer_uses_only_target_frames_for_vq_statistics() -> None:
    quantizer = Quantizer(code_dim=1, codebook_num=1, codebook_size=2)
    layer = quantizer.codebook.layers[0]
    layer.decay = 0.0
    with torch.no_grad():
        layer.embed.copy_(torch.tensor([[0.0, 10.0]]))
        layer.embed_avg.copy_(layer.embed)
    control_quantizer = copy.deepcopy(quantizer)

    prefix = torch.zeros(1, 1, 53)
    target = torch.full((1, 1, 32), 10.0)
    z = torch.cat((prefix, target), dim=-1)

    zq, losses, perplexities = quantizer(z, num_context_frames=53)
    _, control_losses, control_perplexities = control_quantizer(
        target,
        num_context_frames=0,
    )

    assert zq.shape == z.shape
    assert losses.shape == perplexities.shape == (1,)
    torch.testing.assert_close(layer.cluster_size, torch.tensor([0.0, 32.0]))
    torch.testing.assert_close(perplexities, torch.ones(1))
    torch.testing.assert_close(
        layer.cluster_size,
        control_quantizer.codebook.layers[0].cluster_size,
    )
    torch.testing.assert_close(losses, control_losses)
    torch.testing.assert_close(perplexities, control_perplexities)


@pytest.mark.parametrize("num_context_frames", (-1, 5))
def test_quantizer_rejects_invalid_context_frame_count(
    num_context_frames: int,
) -> None:
    quantizer = Quantizer(code_dim=1, codebook_num=1, codebook_size=2)
    z = torch.zeros(1, 1, 5)

    with pytest.raises(ValueError, match="num_context_frames must be in"):
        quantizer(z, num_context_frames=num_context_frames)


class _RecordingGenerator:
    def __init__(self):
        self.calls = []

    def __call__(self, audio, **kwargs):
        self.calls.append((audio.detach().clone(), kwargs))
        scalar = torch.zeros(1)
        return audio + 1, scalar, scalar, scalar, scalar


class _RecordingDiscriminator:
    def __init__(self):
        self.calls = []

    def __call__(self, audio):
        self.calls.append(audio.detach().clone())
        return torch.zeros(1)


def _context_trainer(context_length=3, num_context_frames=1):
    trainer = object.__new__(Trainer)
    trainer.context_length = context_length
    trainer.num_context_frames = num_context_frames
    trainer.device = torch.device("cpu")
    trainer.steps = 0
    trainer.generator_start = 0
    trainer.discriminator_start = 0
    trainer.generator_train = True
    trainer.discriminator_train = True
    trainer.fix_encoder = False
    trainer.paradigm = "standard"
    trainer.config = {"use_feat_match_loss": True}
    trainer.model = {
        "generator": _RecordingGenerator(),
        "discriminator": _RecordingDiscriminator(),
    }
    trainer.tqdm = SimpleNamespace(update=lambda _: None)
    trainer._perplexity = lambda *args, **kwargs: None
    trainer._vq_loss = lambda *args, **kwargs: torch.tensor(0.0)
    trainer._adv_loss = lambda *args, **kwargs: torch.tensor(0.0)
    trainer._dis_loss = lambda *args, **kwargs: torch.tensor(0.0)
    trainer._record_loss = lambda *args, **kwargs: None
    trainer._update_generator = lambda *args, **kwargs: None
    trainer._update_discriminator = lambda *args, **kwargs: None
    trainer._check_train_finish = lambda: False
    return trainer


def test_trainer_accepts_zero_context_without_encoder_strides() -> None:
    assert Trainer._validate_context(0, None) is None


def test_trainer_context_validation_returns_hop_length() -> None:
    assert Trainer._validate_context(15900, (3, 4, 5, 5)) == 300


@pytest.mark.parametrize(
    ("context_length", "enc_strides", "message"),
    (
        (-1, (3, 4, 5, 5), "must be non-negative"),
        (300, None, "enc_strides is required"),
        (301, (3, 4, 5, 5), "divisible by hop length 300"),
    ),
)
def test_trainer_rejects_invalid_context_configuration(
    context_length,
    enc_strides,
    message,
) -> None:
    with pytest.raises(ValueError, match=message):
        Trainer._validate_context(context_length, enc_strides)


def test_autoencoder_losses_and_discriminator_use_only_target_suffix() -> None:
    trainer = _context_trainer()
    metric_calls = []
    trainer._metric_loss = lambda predicted, natural, **kwargs: (
        metric_calls.append((predicted.detach().clone(), natural.detach().clone()))
        or torch.tensor(0.0)
    )
    batch = torch.arange(7, dtype=torch.float).reshape(1, 1, 7)

    trainer._train_step(batch)

    generator = trainer.model["generator"]
    assert len(generator.calls) == 2
    assert all(call.shape[-1] == 7 for call, _ in generator.calls)
    assert all(kwargs == {"num_context_frames": 1} for _, kwargs in generator.calls)
    assert len(metric_calls) == 1
    predicted, natural = metric_calls[0]
    assert predicted.shape[-1] == natural.shape[-1] == 4
    torch.testing.assert_close(natural, batch[..., 3:])
    assert all(call.shape[-1] == 4 for call in trainer.model["discriminator"].calls)


@pytest.mark.parametrize(
    ("context_length", "num_context_frames", "input_length"),
    ((0, 0, 4), (3, 1, 7)),
)
def test_context_and_control_have_the_same_optimizer_update_cadence(
    context_length,
    num_context_frames,
    input_length,
) -> None:
    trainer = _context_trainer(context_length, num_context_frames)
    generator_updates = []
    discriminator_updates = []
    trainer._metric_loss = lambda *args, **kwargs: torch.tensor(0.0)
    trainer._update_generator = generator_updates.append
    trainer._update_discriminator = discriminator_updates.append
    batch = torch.arange(input_length, dtype=torch.float).reshape(1, 1, -1)

    trainer._train_step(batch)

    assert trainer.steps == 1
    assert len(generator_updates) == 1
    assert len(discriminator_updates) == 1


def test_zero_context_passes_zero_frames_and_preserves_audio() -> None:
    trainer = _context_trainer(context_length=0, num_context_frames=0)
    trainer.discriminator_start = 1
    audio = torch.arange(4, dtype=torch.float).reshape(1, 1, 4)

    trainer._train_step(audio)

    generator = trainer.model["generator"]
    assert generator.calls[0][1] == {"num_context_frames": 0}
    assert trainer.trim_context(audio) is audio

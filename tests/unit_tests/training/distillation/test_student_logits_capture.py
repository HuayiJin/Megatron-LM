# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Exercise student-logits capture with Megatron's native cross entropy and KD loss."""

import pytest
import torch
import torch.nn.functional as F

from megatron.core.ops.loss.cross_entropy import unfused_cross_entropy
from megatron.core.ops.loss.fused_cross_entropy import fused_vocab_parallel_cross_entropy
from megatron.training.distillation import cached_logits_loss
from megatron.training.distillation.cached_logits_loss import StudentLogitsCapture, topk_kl_div


class _SingleRankTPGroup:
    def rank(self):
        return 0

    def size(self):
        return 1


class _OutputLayer(torch.nn.Linear):
    """Return the same (logits, bias) pair as ColumnParallelLinear."""

    def forward(self, hidden):
        return super().forward(hidden), None


@pytest.fixture
def capture(monkeypatch):
    monkeypatch.setattr(cached_logits_loss, "_ACTIVE_STUDENT_LOGITS_CAPTURE", None)
    capture = StudentLogitsCapture()
    yield capture
    capture.remove_hooks()


@pytest.mark.parametrize(
    "cross_entropy", [unfused_cross_entropy, fused_vocab_parallel_cross_entropy]
)
def test_capture_preserves_logits_and_joint_lm_kd_gradients(capture, monkeypatch, cross_entropy):
    # A single TP rank has an identity all-reduce. Keep the actual CE forward and backward.
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)
    group = _SingleRankTPGroup()
    generator = torch.Generator().manual_seed(0)
    model = torch.nn.Module()
    model.output_layer = _OutputLayer(16, 32, bias=False)
    with torch.no_grad():
        model.output_layer.weight.copy_(0.1 * torch.randn(32, 16, generator=generator))
    capture.attach_hooks(model)
    hidden = torch.randn(4, 2, 16, generator=generator)
    labels = torch.randint(32, (4, 2), generator=generator)
    teacher_logprobs = torch.randn(4, 2, 32, generator=generator).log_softmax(dim=-1)
    teacher_values, teacher_indices = teacher_logprobs.topk(8, dim=-1)

    logits, _ = model.output_layer(hidden)
    expected = logits.detach().clone()
    lm = cross_entropy(logits, labels, group).sum()
    student_logits = capture.pop()
    torch.testing.assert_close(student_logits.detach(), expected, rtol=0, atol=0)
    assert not torch.equal(logits.detach(), expected)
    kd = topk_kl_div(student_logits, teacher_values, teacher_indices, 1, 0, group).sum()
    (lm + kd).backward()

    ref_weight = model.output_layer.weight.detach().clone().requires_grad_()
    ref_logits = F.linear(hidden, ref_weight)
    ref_lm = F.cross_entropy(ref_logits.flatten(0, 1), labels.flatten(), reduction="sum")
    ref_logprobs = ref_logits.log_softmax(dim=-1).gather(-1, teacher_indices)
    ref_kd = (teacher_values.exp() * (teacher_values - ref_logprobs)).sum()
    (ref_lm + ref_kd).backward()
    torch.testing.assert_close(lm, ref_lm)
    torch.testing.assert_close(kd, ref_kd)
    torch.testing.assert_close(model.output_layer.weight.grad, ref_weight.grad)


def test_capture_is_skipped_in_eval_mode(capture):
    model = torch.nn.Module()
    model.output_layer = _OutputLayer(16, 32, bias=False)
    model.eval()
    capture.attach_hooks(model)
    model.output_layer(torch.zeros(4, 2, 16))

    with pytest.raises(RuntimeError, match="No student logits"):
        capture.pop()

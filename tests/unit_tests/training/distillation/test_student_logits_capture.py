# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""StudentLogitsCapture must hand the KD loss the logits the output layer produced,
even though the model's LM cross entropy consumes those logits in place first."""

import pytest
import torch
import torch.nn.functional as F

from megatron.core.ops.loss.cross_entropy import VocabParallelCrossEntropy
from megatron.training.distillation.cached_logits_loss import StudentLogitsCapture


class _OutputLayer(torch.nn.Module):
    """Returns ``(logits, bias)`` like ``ColumnParallelLinear``."""

    def __init__(self, dtype):
        super().__init__()
        self.linear = torch.nn.Linear(16, 32, bias=False, dtype=dtype)

    def forward(self, hidden):
        return self.linear(hidden), None


class _Model(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        torch.manual_seed(0)
        self.output_layer = _OutputLayer(dtype)


def _consume_like_cross_entropy(logits: torch.Tensor) -> None:
    """What the native CE target does to fp32 logits inside its autograd.Function."""
    with torch.no_grad():
        logits_max = logits.max(dim=-1)[0]
        VocabParallelCrossEntropy.exp_logits_inplace(logits, logits_max)
        logits.div_(logits.sum(dim=-1, keepdim=True))


@pytest.fixture
def capture():
    capture = StudentLogitsCapture()
    yield capture
    capture.remove_hooks()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_captured_logits_survive_in_place_cross_entropy(capture, dtype):
    model = _Model(dtype).train()
    capture.attach_hooks(model)
    hidden = torch.randn(4, 2, 16, dtype=dtype)

    logits, _ = model.output_layer(hidden)
    expected = logits.detach().clone()
    _consume_like_cross_entropy(logits)

    captured = capture.pop()
    assert torch.equal(captured.detach(), expected)
    assert not torch.equal(logits.detach(), expected)


def test_kd_gradient_flows_through_captured_logits(capture):
    model = _Model(torch.float32).train()
    capture.attach_hooks(model)
    hidden = torch.randn(4, 2, 16)
    teacher = torch.randn(4, 2, 32).log_softmax(dim=-1)

    logits, _ = model.output_layer(hidden)
    _consume_like_cross_entropy(logits)
    kd = F.kl_div(capture.pop().log_softmax(dim=-1), teacher, log_target=True, reduction="sum")
    kd.backward()
    grad = model.output_layer.linear.weight.grad.clone()

    model.output_layer.linear.weight.grad = None
    reference = model.output_layer.linear(hidden)
    F.kl_div(reference.log_softmax(dim=-1), teacher, log_target=True, reduction="sum").backward()
    torch.testing.assert_close(grad, model.output_layer.linear.weight.grad)


def test_capture_is_skipped_in_eval_mode(capture):
    model = _Model(torch.float32).eval()
    capture.attach_hooks(model)
    model.output_layer(torch.randn(4, 2, 16))
    with pytest.raises(RuntimeError, match="No student logits"):
        capture.pop()

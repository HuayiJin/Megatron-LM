# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Saver/loader round trip of cached-logits payloads for vocabularies above 2**17."""

import io
from collections import OrderedDict

import pytest
import torch

from megatron.training.distillation import logits_saver
from megatron.training.distillation.logits_saver import LogitsSaverHooks
from megatron.training.distillation.utils import (
    decode_logprobs_payload,
    index_high_bits_for_vocab,
    v2_pack_indices,
)

zstandard = pytest.importorskip("zstandard")


def _bare_saver(**attrs) -> LogitsSaverHooks:
    saver = object.__new__(LogitsSaverHooks)
    saver._pending_writes = OrderedDict()
    saver._index_high_bits = 1
    for name, value in attrs.items():
        setattr(saver, name, value)
    return saver


@pytest.mark.parametrize("vocab_size", [32_000, 151_936, 262_144])
def test_buffered_payload_round_trips_through_decoder(monkeypatch, vocab_size):
    monkeypatch.setattr(logits_saver, "get_consumed_train_samples", lambda: 0)
    monkeypatch.setattr(logits_saver, "get_num_microbatches", lambda: 1)
    saver = _bare_saver(dp_size=1, _micro_batch_size=2)

    torch.manual_seed(0)
    num_high_bits = index_high_bits_for_vocab(vocab_size)
    values = torch.randn(5, 2, 8).half()
    indices = torch.randint(0, vocab_size, (5, 2, 8), dtype=torch.long)
    indices[0, 0, 0] = vocab_size - 1
    indices_low, high_bits = v2_pack_indices(indices, num_high_bits=num_high_bits)
    saver._buffer_iteration(values, indices_low, high_bits, None, index_high_bits=num_high_bits)

    (payload,) = saver._pending_writes.values()
    decoded_values, decoded_indices, cu_seqlens = decode_logprobs_payload(
        zstandard.ZstdCompressor(level=1).compress(payload)
    )
    assert torch.equal(decoded_values, values)
    assert torch.equal(decoded_indices, indices)
    assert cu_seqlens is None

    raw = torch.load(io.BytesIO(payload), weights_only=True)
    if num_high_bits == 1:
        # Vocabularies up to 2**17 keep the original v2 payload keys.
        assert "bit_17" in raw and "index_high_planes" not in raw
    else:
        # Older readers fail on the missing bit_17 key rather than truncating indices.
        assert "bit_17" not in raw and raw["index_high_bits"] == num_high_bits


@pytest.mark.parametrize("vocab_size", [151_936, 262_144])
def test_process_single_microbatch_accepts_vocab_above_2_pow_17(vocab_size):
    saver = _bare_saver(tp_size=1, k=4, p=None, _save_dtype=torch.float16)
    logits = torch.randn(3, 1, vocab_size)
    logits[0, 0, vocab_size - 1] = 100.0

    values, indices = saver._process_single_microbatch(logits)

    assert saver._index_high_bits == index_high_bits_for_vocab(vocab_size)
    assert indices[0, 0, 0].item() == vocab_size - 1
    expected = torch.topk(torch.log_softmax(logits.float(), dim=-1), 4, dim=-1)
    assert torch.equal(indices, expected.indices)
    torch.testing.assert_close(values, expected.values.half())

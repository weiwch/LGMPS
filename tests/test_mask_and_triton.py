from __future__ import annotations

import pytest
import torch

from lgmps.nn.attention import (
    build_session_causal_flex_mask,
    build_session_local_flex_mask,
    segmented_causal_sdpa,
    session_flex_attention,
)
from lgmps.nn.masking import (
    build_session_causal_visibility,
    build_session_visibility,
    segmented_attention_reference,
)
from lgmps.nn.triton_session_attention import session_flash_attention


def test_session_local_visibility_exact():
    item_ids = torch.tensor([0, 0, 1, 1, 2, 2])
    mask = build_session_visibility(item_ids)
    assert mask[4].tolist() == [True, True, False, False, True, False]
    assert mask[5].tolist() == [True, True, False, False, True, True]
    assert not bool(mask[1, 2])
    assert bool(mask[3, 0])


def test_session_causal_visibility_exact():
    item_ids = torch.tensor([0, 1, 2, 0, 1, 1, 2])
    session_ids = torch.tensor([1, 1, 1, 2, 2, 2, 2])
    mask = build_session_causal_visibility(item_ids, session_ids)
    assert mask[4].tolist() == [True, True, True, True, True, False, False]
    assert mask[5].tolist() == [True, True, True, True, True, True, False]
    assert mask[6].tolist() == [True, True, True, True, False, False, True]
    assert not bool(mask[2, 1])


def test_segmented_sdpa_matches_causal_oracle():
    torch.manual_seed(42)
    query = torch.randn(11, 2, 8)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    offsets = torch.tensor([0, 4, 11], dtype=torch.int32)
    actual = segmented_causal_sdpa(query, key, value, offsets)
    expected = segmented_attention_reference(query, key, value, offsets, pattern="causal")
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the Triton kernel")
def test_triton_forward_and_backward_match_reference():
    torch.manual_seed(42)
    query = torch.randn(13, 2, 32, device="cuda", dtype=torch.float16, requires_grad=True)
    key = torch.randn_like(query, requires_grad=True)
    value = torch.randn_like(query, requires_grad=True)
    item_ids = torch.tensor([0, 0, 1, 1, 2, 0, 0, 1, 1, 1, 2, 2, 3], device="cuda")
    offsets = torch.tensor([0, 5, 13], device="cuda", dtype=torch.int32)
    actual = session_flash_attention(query, key, value, item_ids, offsets, use_triton=True)
    expected = segmented_attention_reference(
        query,
        key,
        value,
        offsets,
        item_ids=item_ids,
        pattern="session",
    )
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
    output_gradient = torch.randn_like(actual)
    actual_gradients = torch.autograd.grad(actual, (query, key, value), output_gradient, retain_graph=True)
    expected_gradients = torch.autograd.grad(expected, (query, key, value), output_gradient)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual_gradient, expected_gradient, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for FlexAttention")
def test_flex_training_attention_matches_session_local_oracle():
    torch.manual_seed(13)
    base = [torch.randn(13, 2, 32, device="cuda", dtype=torch.float16) for _ in range(3)]
    query, key, value = [tensor.detach().clone().requires_grad_(True) for tensor in base]
    reference_query, reference_key, reference_value = [
        tensor.detach().clone().requires_grad_(True) for tensor in base
    ]
    item_ids = torch.tensor([0, 0, 1, 1, 2, 0, 0, 1, 1, 1, 2, 2, 3], device="cuda")
    offsets = torch.tensor([0, 5, 13], device="cuda", dtype=torch.int32)
    block_mask = build_session_local_flex_mask(item_ids, offsets)
    assert block_mask is not None

    actual = session_flex_attention(query, key, value, block_mask)
    expected = segmented_attention_reference(
        reference_query,
        reference_key,
        reference_value,
        offsets,
        item_ids=item_ids,
        pattern="session",
    )
    actual.float().square().sum().backward()
    expected.float().square().sum().backward()

    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(query.grad, reference_query.grad, atol=8e-3, rtol=8e-3)
    torch.testing.assert_close(key.grad, reference_key.grad, atol=8e-3, rtol=8e-3)
    torch.testing.assert_close(value.grad, reference_value.grad, atol=8e-3, rtol=8e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for FlexAttention")
def test_flex_training_attention_matches_session_causal_oracle():
    torch.manual_seed(17)
    base = [torch.randn(13, 2, 32, device="cuda", dtype=torch.float16) for _ in range(3)]
    query, key, value = [tensor.detach().clone().requires_grad_(True) for tensor in base]
    reference_query, reference_key, reference_value = [
        tensor.detach().clone().requires_grad_(True) for tensor in base
    ]
    item_ids = torch.tensor([0, 1, 2, 0, 1, 1, 2, 0, 1, 0, 1, 2, 2], device="cuda")
    session_ids = torch.tensor([1, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2], device="cuda")
    offsets = torch.tensor([0, 7, 13], device="cuda", dtype=torch.int32)
    block_mask = build_session_causal_flex_mask(item_ids, session_ids, offsets)
    assert block_mask is not None

    actual = session_flex_attention(query, key, value, block_mask)
    expected = segmented_attention_reference(
        reference_query,
        reference_key,
        reference_value,
        offsets,
        item_ids=item_ids,
        session_ids=session_ids,
        pattern="session_causal",
    )
    actual.float().square().sum().backward()
    expected.float().square().sum().backward()

    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(query.grad, reference_query.grad, atol=8e-3, rtol=8e-3)
    torch.testing.assert_close(key.grad, reference_key.grad, atol=8e-3, rtol=8e-3)
    torch.testing.assert_close(value.grad, reference_value.grad, atol=8e-3, rtol=8e-3)

"""Attention layers."""

import functools
import math
import string
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

from absl import logging
from flax import linen as nn
import jax
from jax import numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from einops import rearrange, repeat  # XD
import numpy as np
from praxis import asserts
from praxis import base_layer
from praxis import pax_fiddle
from praxis import py_utils
from praxis import pytypes
from praxis.layers import base_ops
from praxis.layers import embedding_softmax
from praxis.layers import stochastics
from praxis.layers import normalizations  # XD

NestedMap = py_utils.NestedMap
WeightInit = base_layer.WeightInit
WeightHParams = base_layer.WeightHParams
template_field = base_layer.template_field
LayerTpl = pax_fiddle.Config[base_layer.BaseLayer]
JTensor = pytypes.JTensor
NestedJTensor = pytypes.NestedJTensor
NestedInt = pytypes.NestedInt

SplitDimsMapping = pytypes.SplitDimsMapping

class DotProductAttention(base_layer.BaseLayer):
  def _atten_context(
      self,
      query: JTensor,
      key: JTensor,
      value: JTensor,
      atten_mask: JTensor,
  ) -> Tuple[JTensor, JTensor]:
    # logits = self._atten_logits(query, key)
    logits = self.qk_einsum(f"BNTH,BNSH->BNTS", query, key)  # XD

    if self.scale_logits_by_head_dims:
      logits = jnp.multiply(logits, 1.0 / np.sqrt(query.shape[-1]))

    logits = self._cap_logits(logits)
    logits = logits.astype(jnp.float32)
    padded_logits = py_utils.apply_mask_to_logits(logits, atten_mask)
    if self.attention_extra_logit is None:
      probs = jax.nn.softmax(padded_logits, axis=-1).astype(value.dtype)
    else:
      probs = jnp.exp(self._log_softmax_with_extra_logit(padded_logits)).astype(value.dtype)

    probs = self.atten_dropout(probs)
    encoded = self.pv_einsum(f'BNTS,BNSH->BNTH', probs, value)
    return encoded, probs
    
  def _dot_atten(
      self,
      query: JTensor,
      key: JTensor,
      value: JTensor,
      atten_mask: JTensor,
  ) -> Tuple[JTensor, JTensor]:
    """Main attention function.

    Args:
      query: JTensor of shape [B, T, N, H].
      key: JTensor of shape [B, S, N, H].
      value: JTensor of shape [B, S, N, H].
      atten_mask: JTensor of shape [1|B, 1, 1|T, S] which is a mask that is
        applied to prevent attention between unwanted pairs. This has already
        been converted into large negative logits. Note that the first and third
        dimension allow size 1 if the mask is shared by every item in the batch
        or every token in the target sequence.
      relative_bias: Relative bias of shape [B, N, T, S].

    Returns:
      encoded: JTensor of shape [B, T, N, H].
      atten_probs: JTensor of shape [B, N, T, S].
    """
    query = self._shard_blnh(query)
    key = self._shard_blnh(key)
    value = self._shard_blnh(value)

    b, t, n, h = query.shape
    s = key.shape[1]
    # base_layer.assert_has_shape(value, [b, s, n, -1])  # XD: h -> -1
    # base_layer.assert_has_shape(query, [b, -1, n, h])
    # If only padding bias is supplied, then atten_mask can be [B, 1, 1, S]
    # since each target token is prohibited from attending to the same set of
    # source tokens. In this case tiling is inefficient and unnecessary.
    # If there is no padding mask, and only causal mask then the shape can be
    # [1, 1, T, S]
    base_layer.assert_has_shape(atten_mask, [-1, 1, -1, s])
    asserts.in_set(atten_mask.shape[2], [t, 1])
    asserts.in_set(atten_mask.shape[0], [b, 1])

    query = self._scale_query(query)
    query, key, value = [tensor.transpose(0, 2, 1, 3) for tensor in [query, key, value]] # btnh->bnth
    if self.query_chunk_size is None:
      encoded, probs = self._atten_context(query, key, value, atten_mask)
    else:
      w = self.query_chunk_size
      assert t % w == 0, f'{t} % {w} != 0'
      encoded = jnp.zeros((b, n, t, h), dtype=value.dtype)
      for i in range(t // w):
        start, stop = i * w, (i + 1) * w
        _query = query[:, :, start : stop, :]
        _key, _value = key[:, :, : stop, :], value[:, :, : stop, :]
        _atten_mask = atten_mask[:, :, start : stop, : stop]
        _encoded, _ = self._atten_context(_query, _key, _value, _atten_mask)
        encoded = encoded.at[:, :, start : stop, :].set(_encoded)
    encoded = encoded.transpose(0, 2, 1, 3)  # bnth->btnh


    # logits = self._atten_logits(query, key)
    # if relative_bias is not None:
    #   # The relative_bias has shape [1, n, t, s] or [b, n, t, s].
    #   base_layer.assert_has_shape(relative_bias, [-1, n, t, s])
    #   logits += relative_bias
    # logits = checkpoint_name(logits, 'logits')

    # if self.scale_logits_by_head_dims:
    #   logits = jnp.multiply(logits, 1.0 / np.sqrt(h))

    # self.add_summary(
    #     'max_logit_precap',
    #     jnp.max(py_utils.apply_mask_to_logits(logits, atten_mask)),
    #     verbosity=4,
    # )
    # self.add_summary(
    #     'rms_logits_precap',
    #     ((logits**2.0).mean().astype(jnp.float32) ** 0.5),
    #     verbosity=4,
    # )
    # logits = self._cap_logits(logits)
    # # Attention softmax is always carried out in fp32.
    # logits = logits.astype(jnp.float32)
    # # Apply attention masking
    # padded_logits = py_utils.apply_mask_to_logits(logits, atten_mask)
    # if self.attention_mask_summary:
    #   self.add_summary('attention_mask', atten_mask)
    # if self.attention_extra_logit is None:
    #   probs = jax.nn.softmax(padded_logits, axis=-1).astype(value.dtype)
    # else:
    #   probs = jnp.exp(self._log_softmax_with_extra_logit(padded_logits)).astype(value.dtype)

    # # Apply attention dropout.
    # probs = self.atten_dropout(probs)
    # # Compute the attention context.
    # if self.transpose_logits: probs = jnp.transpose(probs, (0, 3, 1, 2)) # XD: BTSN -> BNTS
    # encoded = self.pv_einsum(f'BNTS,BSNH->BTNH', probs, value)

    # if self.zero_fully_masked:
    #   # Return zeros for tokens which don't attend anything.
    #   fully_masked = jnp.all(
    #       atten_mask < py_utils.get_large_negative_number(jnp.float32) / 2,
    #       axis=-1,
    #   )[:, 0, :, jnp.newaxis, jnp.newaxis]
    #   encoded *= 1 - fully_masked


    encoded = checkpoint_name(encoded, 'context')
    encoded = self._shard_blnh(encoded)
    return encoded, probs if self.query_chunk_size is None else None
# coding=utf-8
# Copyright 2022 The Pax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Embedding and softmax layers."""

import math
from typing import Optional, Union
from absl import logging  # XD

import jax
from jax import numpy as jnp
import numpy as np
from praxis import base_layer
from praxis import pax_fiddle
from praxis import py_utils
from praxis import pytypes
from praxis.layers import activations
from praxis.layers import base_ops
from praxis.layers import linears

NestedMap = py_utils.NestedMap
WeightHParams = base_layer.WeightHParams

JTensor = pytypes.JTensor

SplitDimsMapping = pytypes.SplitDimsMapping
LayerTpl = pax_fiddle.Config[base_layer.BaseLayer]

template_field = base_layer.template_field


def _compute_z_loss(logits):
  """Returns a z_loss regularization which stablize logits."""
  # Applies stop_gradient to max_logit instead of logits.
  max_logit = jax.lax.stop_gradient(jnp.max(logits, axis=-1, keepdims=True))
  exp_x = jnp.exp(logits - max_logit)
  sum_exp_x = jnp.sum(exp_x, axis=-1, keepdims=True)
  log_z = jnp.log(sum_exp_x) + max_logit
  return jnp.square(log_z)


class TokenCounter(base_layer.BaseLayer):
  """Keep track of total tokens seen during training."""

  def setup(self) -> None:
    """Creates non-trainable counter."""
    # Currently, uint64 support is explicitly disabled as it causes
    # causes different type promotion behaviors. NB: Using a uint64 type
    # will fall-back to a uint32 with a UserWarning.
    #
    # Using float32 approximation, tracking millions of tokens.
    # Do not allow bfloat16 conversion.
    approx_total_tokens_mm = WeightHParams(
        shape=(),
        init=base_layer.WeightInit.Constant(0.0),
        collections=[
            base_layer.WeightHParamsCollection.REQUIRES_SUM_SYNC,
            base_layer.WeightHParamsCollection.DISALLOW_BFLOAT16_CONVERSION,
        ],
    )
    self.create_variable(
        'approx_total_tokens_mm', approx_total_tokens_mm, trainable=False
    )

  def __call__(self, inputs: JTensor, paddings: JTensor) -> JTensor:
    """Track total non-padding tokens.

    Args:
      inputs: Input ids. An int32 JTensor of shape [B, T].
      paddings: A 0/1 JTensor of shape [B, T] with 1 denoting padding.
    """
    if not self.do_eval:
      approx_total_tokens_mm = self.get_var('approx_total_tokens_mm')
      self.add_summary('approx_total_tokens_mm', approx_total_tokens_mm)
      scale = jnp.array(1e6, dtype=jnp.float32)
      batch_total_mm = jnp.sum(1.0 - paddings).astype(jnp.float32) / scale
      # Force f32 addition.
      new_approx_total_tokens_mm = approx_total_tokens_mm.astype(
          jnp.float32
      ) + batch_total_mm.astype(jnp.float32)
      self.update_var('approx_total_tokens_mm', new_approx_total_tokens_mm)  # pytype: disable=bad-return-type  # jax-ndarray


class Embedding(base_layer.BaseLayer):
  """A simple embedding layer that performs embedding lookups from ids.

  Attributes:
    num_classes: Number of tokens in the vocabulary.
    input_dims: Depth of the embedding output. This is called `input_dims` as
      opposed to the more appropriate `embedding_dims` to be compatible with
      other softmax/embedding layers defined in this file.
    lookup_style: Style of lookup, one of index or matmul.
    scale_sqrt_depth: If set to True, activations are scaled with
      sqrt(embedding_dim) in emb_lookup.
    set_nan_for_oob_id: If set to True, embeddings corresponding to
      out-of-boundaries ids will be set to NaN. Useful for debugging purposes.
  """

  num_classes: int = 0
  input_dims: int = 0
  lookup_style: str = 'index'
  scale_sqrt_depth: bool = False
  set_nan_for_oob_id: bool = False
  array_lookup_tpl: LayerTpl = template_field(base_ops.ArrayLookup)
  einsum_tpl: LayerTpl = template_field(base_ops.EinsumOp)

  class ActivationSharding(base_layer.BaseLayer.ActivationSharding):
    """Represents how intermediate values should be partitioned across a mesh.

    Attributes:
      emb_out_split_dims_mapping: Sharding of the emb output.
    """

    emb_out_split_dims_mapping: SplitDimsMapping = None

  def setup(self) -> None:
    assert self.num_classes > 0
    assert self.input_dims > 0

    wp = self.weight_split_dims_mapping
    self.create_variable(
        'emb_var',
        WeightHParams(
            shape=[self.num_classes, self.input_dims],
            mesh_shape=self.mesh_shape,
            tensor_split_dims_mapping=wp.wt,
        ),
    )
    self.create_child('array_lookup', self.array_lookup_tpl.clone())
    self.create_child('einsum', self.einsum_tpl.clone())

  def __call__(self, ids: JTensor) -> JTensor:
    return self.emb_lookup(ids)

  def emb_lookup(self, ids: JTensor) -> JTensor:
    ap = self.activation_split_dims_mapping

    if self.lookup_style == 'index':
      embs = self.array_lookup(jnp.asarray(self.theta.emb_var), (ids,))
    elif self.lookup_style == 'matmul':
      # Explicit casting to fprop_dtype needed for bf16.
      one_hot_ids = jax.nn.one_hot(
          ids, self.num_classes, dtype=self.fprop_dtype
      )
      embs = self.einsum('...y,yz->...z', one_hot_ids, self.theta.emb_var)
    else:
      raise ValueError('Unknown lookup style.')

    # map out-of-boundary ids to nan for easier debug
    if self.set_nan_for_oob_id:
      embs = jnp.where(ids[..., jnp.newaxis] < self.num_classes, embs, jnp.nan)

    if self.scale_sqrt_depth:
      embs *= self.input_dims**0.5

    embs = base_layer.maybe_shard(
        embs, ap.emb_out_split_dims_mapping, self.mesh_axis_names
    )
    return embs

  def extend_step(self, ids: JTensor, *, time_step: JTensor) -> JTensor:
    del time_step  # Not used.
    return self.emb_lookup(ids)


class FullSoftmax(base_layer.BaseLayer):
  """A simple softmax layer with cross-entropy outputs.

  Attributes:
    input_dims: Dimension of the input when used as a softmax layer. This is
      also the depth of the output when used as an embedding layer.
    num_classes: Total number of target classes when used as a softmax layer.
      This is also the size of the vocabulary when used as an embedding layer.
    soft_cap_logits: If not None logits are soft capped to this value.
    bi_tempered_loss: If not None applies bi-tempered loss.
    label_smoothing_prob: Label smoothing probability.
    label_smoothing_apply_for_eval: If False, disables label smoothing at eval
      time, even if p.label_smoothing_prob > 0. Label smoothing is a form of
      regularization and we may want to disable it at eval time.
    z_loss_weight: If z_loss_weight is nonzero, we add a loss equal to
      z_loss_weight * square(logsumexp(logits, -1))
    bias_init: Init scale (constant) of bias terms.
    feed_forward_tpl: Sub configurable field for the feed-forward layer. If
      None, skip feedforward layer and directly apply softmax to the input.
  """

  input_dims: int = 0
  num_classes: int = 0
  soft_cap_logits: Optional[float] = 0.0
  bi_tempered_loss_tpl: Optional[LayerTpl] = template_field(None)
  label_smoothing_prob: float = 0.0
  label_smoothing_apply_for_eval: bool = True
  z_loss_weight: float = 0.0
  bias_init: Optional[float] = 0.0
  feed_forward_tpl: LayerTpl = template_field(linears.FeedForward)
  chunk_size: int = None  # XD
  loss_batch_mean: bool = False

  def setup(self) -> None:
    if self.feed_forward_tpl is not None:
      wp = self.weight_split_dims_mapping
      ap = self.activation_split_dims_mapping
      ff_p = self.feed_forward_tpl.clone().set(
          input_dims=self.input_dims,
          output_dims=self.num_classes,
          activation_tpl=pax_fiddle.Config(activations.Identity),
          bias_init=self.bias_init,
          weight_split_dims_mapping=wp.clone(),
          activation_split_dims_mapping=ap.clone(),
      )
      self.create_child('logits_ffn', ff_p)
    if self.bi_tempered_loss_tpl:
      self.create_child('bi_tempered_loss', self.bi_tempered_loss_tpl)

  def get_logits(self, inputs: JTensor) -> JTensor:
    """Returns logits given the inputs with an option to soft cap it.

    Args:
      inputs: a single JTensor with shape [..., input_dim].

    Returns:
      logits: with shape [..., num_classes]. Unnormalized softmax's logits.
    """
    if self.feed_forward_tpl is not None:
      # Compute logits.
      logits = self.logits_ffn(inputs)
    else:
      logits = inputs

    # Soft cap logits if applicable.
    if self.soft_cap_logits:
      logits = self.soft_cap_logits * jnp.tanh(logits / self.soft_cap_logits)
    return logits

  def logits_to_logp(self, logits: JTensor) -> JTensor:
    """Converts logits to log probability scores."""
    return jax.nn.log_softmax(logits)

  def _compute_logits_and_loss(self, inputs, class_weights,
                              class_ids=None, class_probabilities=None):  # XD
    # Compute logits
    logits = self.get_logits(inputs)
    # We perform softmax in float32 to improve stability.
    logits = logits.astype(jnp.float32)
    log_probs = jax.nn.log_softmax(logits)

    if class_probabilities is None:
      class_probabilities = jax.lax.stop_gradient(jax.nn.one_hot(
          jnp.squeeze(class_ids, axis=-1), self.num_classes, dtype=jnp.float32))

    if self.bi_tempered_loss_tpl is None:
      per_example_xent = -jnp.sum(
        log_probs * class_probabilities, axis=-1, dtype=jnp.float32)
    else:
      per_example_xent = self.bi_tempered_loss(logits, class_probabilities)
    per_example_argmax = jax.lax.stop_gradient(
        jnp.argmax(logits.astype(jnp.float32), axis=-1))

    if self.loss_batch_mean:
        # lsp class_weights: bsz * len * 1
        # logging.info(f'============loss_batch_mean: {self.loss_batch_mean}==========')
        logging.info(f'per_example_xent: {per_example_xent.shape} class_weights： {class_weights.shape}' )
        total_xent_batch = jnp.sum(
            jnp.expand_dims(per_example_xent, axis=-1) * class_weights,
            dtype=jnp.float32, axis=-2,
        )
        total_weight_batch = jnp.maximum(jnp.sum(class_weights, dtype=jnp.float32, axis=-2), 1e-10)
        batch_avg_xent = jnp.mean(total_xent_batch / total_weight_batch)
    else:
        # logging.info(f'============loss_batch_mean: {self.loss_batch_mean}==========')
        batch_avg_xent = None

    # Compute total softmax cross-entropy loss for the output tensor.
    total_xent = jnp.sum(
        jnp.expand_dims(per_example_xent, axis=-1) * class_weights, dtype=jnp.float32)
    total_weight = jnp.sum(class_weights, dtype=jnp.float32)

    z_loss = None
    if self.z_loss_weight > 0.0:
      z_loss = (jnp.sum(_compute_z_loss(logits) * class_weights, dtype=jnp.float32)
          / total_weight) * self.z_loss_weight
      # self.add_summary('aux_z_loss', z_loss)
      # self.add_aux_loss('aux_z_loss', z_loss)
    return logits, log_probs, per_example_xent, per_example_argmax, total_xent, total_weight, z_loss, batch_avg_xent

  def __call__(
      self,
      inputs: JTensor,
      class_weights: JTensor,
      class_ids: Optional[JTensor] = None,
      class_probabilities: Optional[JTensor] = None,
  ) -> NestedMap:
    # pyformat:disable
    """Computes logits, softmax cross entropy etc.

    Args:
      inputs:        [..., input_dims].
      class_weights: [..., 1], weights for each target word.
      class_ids:     [..., 1], int32 type, target labels.
      class_probabilities: [..., num_classes].

    Returns:
      A `.NestedMap` containing the following fields

      - logits:    [..., num_classes], unnormalized softmax logits.
      - log_probs: [..., num_classes], normalized softmax logits.
      - per_example_argmax: [...]. argmax of i-th example.
      - per_example_xent:   [...]. Cross entropy between i-th example's
        prediction and its label.
      - per_example_weight: [...]. class_weights casted to this layer's dtype.
      - total_xent:   a scalar, the sum of per_example_weight * per_example_xent.
      - total_weight: a scalar, the sum of per_example_weight.
      - avg_xent: A scalar. total_loss / total_weight.
      - z_loss: (optional) a scalar, the square of logsum logits when
        z_loss_weight > 0.
    """
    # pyformat:enable

    # Assert one of class_ids or class_probabilities is not None
    if class_ids is None and class_probabilities is None:
      raise ValueError('One of class_ids or class_probabilities must be given.')
    inputs_dtype = inputs.dtype

    if self.chunk_size is not None:
      w = self.chunk_size; t = inputs.shape[-2]
      assert t % w == 0, f'{t} % {w} != 0'
      per_example_xent, per_example_argmax = [], []
      logits, log_probs = None, None
      total_xent, total_weight, z_loss, batch_avg_xent = None, None, None, None
      for i in range(t // w):
        start, stop = i * w, (i + 1) * w
        _, _, _per_example_xent, _per_example_argmax, _total_xent, _total_weight, _z_loss, _batch_avg_xent = \
          self._compute_logits_and_loss(inputs[..., start : stop, :],
            class_weights[..., start : stop, :], class_ids[..., start : stop, :])
        per_example_xent.append(_per_example_xent)
        per_example_argmax.append(_per_example_argmax)
        total_xent = _total_xent if total_xent is None else total_xent + _total_xent
        # lsp
        batch_avg_xent = _batch_avg_xent if batch_avg_xent is None else batch_avg_xent + _batch_avg_xent
        total_weight = _total_weight if total_weight is None else total_weight + _total_weight
        z_loss = _z_loss if z_loss is None else z_loss + _z_loss
      # lsp
      batch_avg_xent = batch_avg_xent if batch_avg_xent is None else batch_avg_xent / (i + 1)
      per_example_xent = jnp.concatenate(per_example_xent, axis=-1)
      per_example_argmax = jnp.concatenate(per_example_argmax, axis=-1)
    else:
      logits, log_probs, per_example_xent, per_example_argmax, total_xent, total_weight, z_loss, batch_avg_xent = \
        self._compute_logits_and_loss(inputs, class_weights, class_ids)
       
    output_nmap = NestedMap(
        logits=logits.astype(inputs_dtype) if logits is not None else None,
        log_probs=log_probs.astype(inputs_dtype) if log_probs is not None else None,
        per_example_argmax=per_example_argmax,
        per_example_xent=per_example_xent.astype(jnp.float32),
        total_xent=total_xent,
        total_weight=total_weight,
        avg_xent=(total_xent / (total_weight + 1e-6)).astype(jnp.float32) if not self.loss_batch_mean else batch_avg_xent,
    )
    if self.z_loss_weight > 0.0:
      output_nmap['z_loss'] = z_loss
    return output_nmap
       
    # Compute logits
    inputs_dtype = inputs.dtype
    logits = self.get_logits(inputs)
    # We perform softmax in float32 to improve stability.
    logits = logits.astype(jnp.float32)
    log_probs = jax.nn.log_softmax(logits)

    if class_probabilities is None:
      class_probabilities = jax.nn.one_hot(
          jnp.squeeze(class_ids, axis=-1), self.num_classes, dtype=jnp.float32
      )
      if self.label_smoothing_prob > 0.0:
        # Label smoothing reduce the probability of the label from 1 to
        # 1 - label_smoothing_prob, and redistribute label_smoothing_prob to the
        # rest of num_classes - 1 classes where each class has a probability of
        # label_smoothing_prob / (num_classes - 1).
        if not self.do_eval or self.label_smoothing_apply_for_eval:
          # We may want to disable label smoothing at eval time.
          other_prob = self.label_smoothing_prob / (self.num_classes - 1)
          class_probabilities = (
              (1.0 - self.label_smoothing_prob) * class_probabilities
              + other_prob * (1.0 - class_probabilities)
          ).astype(jnp.float32)
      class_probabilities = jax.lax.stop_gradient(class_probabilities)

    if self.bi_tempered_loss_tpl is None:
      per_example_xent = -jnp.sum(
          log_probs * class_probabilities, axis=-1, dtype=jnp.float32
      )
    else:
      per_example_xent = self.bi_tempered_loss(logits, class_probabilities)
    per_example_argmax = jax.lax.stop_gradient(
        jnp.argmax(logits.astype(jnp.float32), axis=-1)
    )

    # Compute total softmax cross-entropy loss for the output tensor.
    total_xent = jnp.sum(
        jnp.expand_dims(per_example_xent, axis=-1) * class_weights,
        dtype=jnp.float32,
    )
    total_weight = jnp.sum(class_weights, dtype=jnp.float32)

    if self.z_loss_weight > 0.0:
      z_loss = (
          jnp.sum(_compute_z_loss(logits) * class_weights, dtype=jnp.float32)
          / total_weight
      )
      z_loss *= self.z_loss_weight
      self.add_summary('aux_z_loss', z_loss)
      self.add_aux_loss('aux_z_loss', z_loss)

    output_nmap = NestedMap(
        logits=logits.astype(inputs_dtype),
        log_probs=log_probs.astype(inputs_dtype),
        per_example_argmax=per_example_argmax,
        per_example_xent=per_example_xent.astype(jnp.float32),
        total_xent=total_xent,
        total_weight=total_weight,
        avg_xent=(total_xent / (total_weight + 1e-6)).astype(jnp.float32),
    )
    if self.z_loss_weight > 0.0:
      output_nmap['z_loss'] = z_loss
    return output_nmap


class SharedEmbeddingSoftmax(FullSoftmax):
  """A softmax layer that also supports embedding lookups.

  Attributes:
    lookup_style: Style of lookup, one of index or matmul.
    scale_sqrt_depth: If set True, activations are scaled with
      sqrt(embedding_dim) in emb_lookup.
  """

  lookup_style: str = 'index'
  scale_sqrt_depth: bool = False
  array_lookup_tpl: LayerTpl = template_field(base_ops.ArrayLookup)
  einsum_tpl: LayerTpl = template_field(base_ops.EinsumOp)

  def setup(self) -> None:
    super().setup()
    self.create_child('array_lookup', self.array_lookup_tpl.clone())
    self.create_child('einsum', self.einsum_tpl.clone())

  class ActivationSharding(base_layer.BaseLayer.ActivationSharding):
    """Represents how intermediate values should be partitioned across a mesh.

    Attributes:
      emb_out_split_dims_mapping: Sharding of the emb output.
    """

    emb_out_split_dims_mapping: SplitDimsMapping = None

  def emb_lookup(self, ids: JTensor) -> JTensor:
    ap = self.activation_split_dims_mapping
    emb_var = jnp.transpose(self.logits_ffn.linear.theta.w)
    if self.lookup_style == 'index':
      embs = self.array_lookup(jnp.asarray(emb_var), (ids,))
    elif self.lookup_style == 'matmul':
      # Explicit casting to fprop_dtype needed for bf16.
      one_hot_ids = jax.nn.one_hot(
          ids, self.num_classes, dtype=self.fprop_dtype
      )
      embs = self.einsum('...y,yz->...z', one_hot_ids, emb_var)
    else:
      raise ValueError('Unknown lookup style.')
    # Scale with sqrt(embedding dims)
    if self.scale_sqrt_depth:
      embs *= self.input_dims**0.5

    embs = base_layer.maybe_shard(
        embs, ap.emb_out_split_dims_mapping, self.mesh_axis_names
    )
    return embs

  def extend_step(self, ids: JTensor, *, time_step: JTensor) -> JTensor:
    del time_step  # Not used.
    return self.emb_lookup(ids)


class SigmoidCrossEntropy(base_layer.BaseLayer):
  """A sigmoid cross-entropy loss layer with logits projection.

  Attributes:
    input_dims: Dimension of the input when used as a softmax layer. This is
      also the depth of the output when used as an embedding layer.
    num_classes: Total number of target classes when used as a softmax layer.
      This is also the size of the vocabulary when used as an embedding layer.
    soft_cap_logits: If not None logits are soft capped to this value.
    bias_init: Init scale (constant) of bias terms.
    feed_forward_tpl: Sub configurable field for the feed-forward layer. If
      None, skip the FFN projection.
  """

  input_dims: int = 0
  num_classes: int = 0
  soft_cap_logits: Optional[float] = 0.0
  bias_init: Optional[float] = 0.0
  feed_forward_tpl: LayerTpl = template_field(linears.FeedForward)

  def setup(self) -> None:
    if self.feed_forward_tpl:
      wp = self.weight_split_dims_mapping
      ap = self.activation_split_dims_mapping
      ff_p = self.feed_forward_tpl.clone().set(
          input_dims=self.input_dims,
          output_dims=self.num_classes,
          activation_tpl=pax_fiddle.Config(activations.Identity),
          bias_init=self.bias_init,
          weight_split_dims_mapping=wp.clone(),
          activation_split_dims_mapping=ap.clone(),
      )
      self.create_child('logits_ffn', ff_p)
    else:
      if self.input_dims != self.num_classes:
        raise ValueError(
            f"SigmoidCrossEntropy's input_dims {self.input_dims} has to match "
            'num_classes if ffn is disabled.'
        )

  def get_logits(self, inputs: JTensor) -> JTensor:
    """Returns logits given the inputs with an option to soft cap it.

    Args:
      inputs: a single JTensor with shape [..., input_dim].

    Returns:
      logits: with shape [..., num_classes]. Unnormalized softmax's logits.
    """
    if not self.feed_forward_tpl:
      return inputs
    # Compute logits.
    logits = self.logits_ffn(inputs)

    # Soft cap logits if applicable.
    if self.soft_cap_logits:
      logits = self.soft_cap_logits * jnp.tanh(logits / self.soft_cap_logits)
    return logits

  def __call__(
      self,
      inputs: JTensor,
      class_weights: JTensor,
      class_ids: Optional[JTensor] = None,
      class_probabilities: Optional[JTensor] = None,
  ) -> NestedMap:
    """Computes logits, sigmoid cross entropy etc.

    Args:
      inputs: a single JTensor with shape [..., input_dim].
      class_weights: a JTensor with shape [..., 1] or [..., num_classes]
        containing the weights for each target or for each label class.
      class_ids: a JTensor with shape [..., 1] of int32 dtype containing the
        target class labels.
      class_probabilities: a JTensor with shape [..., num_classes] of float
        values indicating class-membership probabilities.

    Returns:
      A `.NestedMap` containing the following fields

      - logits: with shape [..., num_classes]. Unnormalized softmax's logits.
      - log_probs: with shape [..., num_classes]. Log prob of logits.
      - per_example_argmax: with shape [...]. argmax of i-th example.
      - per_example_xent: with shape [...]. Cross entropy between i-th example's
        prediction and its label multiplied by per_class_weight.
      - total_xent: A scalar. The sum of per_example_weight * per_example_xent.
      - total_weight: A scalar. The sum of per_example_weight.
      - avg_xent: A scalar. total_loss / total_weight.
    """
    # Assert one of class_ids or class_probabilities is not None
    if class_ids is None and class_probabilities is None:
      raise ValueError('One of class_ids or class_probabilities must be given.')

    # Compute logits
    inputs_dtype = inputs.dtype
    logits = self.get_logits(inputs)
    # We perform softmax in float32 to improve stability.
    logits = logits.astype(jnp.float32)
    log_probs = jax.nn.log_sigmoid(logits)

    if class_probabilities is None:
      if self.num_classes == 1:
        raise ValueError(
            'one_hot with num_classes=1 has a strange behavior. Please double '
            'check this is what you intended to do.'
        )
      class_probabilities = jax.nn.one_hot(
          jnp.squeeze(class_ids, axis=-1), self.num_classes, dtype=jnp.float32
      )
      class_probabilities = jax.lax.stop_gradient(class_probabilities)

    if class_weights.shape[-1] == self.num_classes:
      per_class_weight = class_weights
      per_example_weight = jnp.ones(class_weights.shape[:-1] + (1,))
    elif class_weights.shape[-1] == 1:
      per_class_weight = jnp.ones(
          class_weights.shape[:-1] + (self.num_classes,)
      )
      per_example_weight = class_weights
    else:
      raise ValueError(
          f'Wrong shape of class_weights {class_weights.shape} vs '
          f'logits shape {logits.shape} vs '
          f'class_probabilities shape {class_probabilities.shape} vs '
          f'num_classes {self.num_classes}'
      )

    # A stable implementation of sigmoid cross entropy loss.
    zeros = jnp.zeros_like(logits)
    cond = logits > zeros
    relu_logits = jnp.where(cond, logits, zeros)
    neg_abs_logits = jnp.where(cond, -logits, logits)
    per_class_xent = (
        relu_logits
        - logits * class_probabilities
        + jnp.log1p(jnp.exp(neg_abs_logits))
    )
    per_example_xent = jnp.sum(
        per_class_xent * per_class_weight, axis=-1, dtype=jnp.float32
    )

    per_example_argmax = jax.lax.stop_gradient(
        jnp.argmax(logits.astype(jnp.float32), axis=-1)
    )

    # Compute total sigmoid cross-entropy loss for the output tensor.
    total_xent = jnp.sum(
        jnp.expand_dims(per_example_xent, axis=-1) * per_example_weight,
        dtype=jnp.float32,
    )

    total_weight = jnp.sum(class_weights, dtype=jnp.float32)
    output_nmap = NestedMap(
        logits=logits.astype(inputs_dtype),
        log_probs=log_probs.astype(inputs_dtype),
        per_example_argmax=per_example_argmax,
        per_example_xent=per_example_xent.astype(inputs_dtype),
        total_xent=total_xent.astype(inputs_dtype),
        total_weight=total_weight,
        avg_xent=(total_xent / (total_weight + 1e-6)).astype(inputs_dtype),
    )
    return output_nmap


class GShardSharedEmbeddingSoftmax(base_layer.BaseLayer):
  """Softmax layer with embedding lookup and Gaussian init used in gshard.

  Features:
  1) Weight shape is [V, M] where V is num_classes and M is input_dims.
  2) No bias
  3) Apply 1/sqrt(M) to the input activations before computing the logits.
  4) Optionally using soft clipping and absolute value clipping of logits.
  5) Optional label smoothing.

  Attributes:
    input_dims: Dimension of the input.
    num_classes: Total number of target classes.
    use_tgt_labels_size_as_loss_denominator: False to use total number of
      non-padding tokens instead of fixed tgt_labels tensor size.
    soft_cap_logits: If not None logits are soft capped to this value before the
      absolute value clipping with p.logits_abs_max.
    logits_abs_max: Absolute logits clipping.
    z_loss_weight: If z_loss_weight is nonzero, we add a loss equal to
      z_loss_weight * square(logsumexp(logits, -1))
    label_smoothing_prob: Optional label smoothing.
  """

  input_dims: int = 0
  num_classes: int = 0
  use_tgt_labels_size_as_loss_denominator: bool = True
  soft_cap_logits: Optional[float] = 0.0
  logits_abs_max: Optional[float] = 0.0
  z_loss_weight: float = 0.0
  label_smoothing_prob: float = 0.0

  class ActivationSharding(base_layer.BaseLayer.ActivationSharding):
    """Represents how intermediate values should be partitioned across a mesh.

    Attributes:
      emb_out_split_dims_mapping: Mesh split for embedding outputs..
    """

    emb_out_split_dims_mapping: SplitDimsMapping = None

  def setup(self) -> None:
    wp = self.weight_split_dims_mapping
    ap = self.activation_split_dims_mapping
    emb_p = pax_fiddle.Config(
        linears.Linear,
        input_dims=self.num_classes,
        output_dims=self.input_dims,
        # Same as in gshard_builder.DenseBuilder.Embedding
        params_init=base_layer.WeightInit.Gaussian(),
        weight_split_dims_mapping=wp.clone(),
        activation_split_dims_mapping=ap.clone(),
    )
    self.create_child('embedding', emb_p)

  def emb_lookup(self, ids: JTensor) -> JTensor:
    ap = self.activation_split_dims_mapping
    # BL -> BLV
    one_hot_ids = jax.nn.one_hot(ids, self.num_classes, dtype=self.fprop_dtype)
    # BLV,VH -> BLH
    embs = linears.project_last_dim(one_hot_ids, self.embedding.theta.w)
    embs = base_layer.maybe_shard(
        embs, ap.emb_out_split_dims_mapping, self.mesh_axis_names
    )
    return embs

  def get_logits(self, inputs: JTensor) -> JTensor:
    """Returns logits given the inputs with an option to cap it.

    Args:
      inputs: a single JTensor with shape [..., input_dim].

    Returns:
      logits: with shape [..., num_classes]. Unnormalized softmax's logits.
    """
    ap = self.activation_split_dims_mapping
    # activations are scaled with 1/sqrt(input_dims)
    inputs *= self.input_dims**-0.5
    # VH -> HV
    softmax_var = jnp.transpose(self.embedding.theta.w)
    # Compute logits:  BLH,HV -> BLV
    logits = linears.project_last_dim(inputs, softmax_var)
    # Adjust sharding annotation during decoding.
    ap_out = ap.out
    if ap_out is not None and len(ap_out) == 3 and logits.ndim == 2:
      ap_out = [ap_out[0], ap_out[2]]
    logits = base_layer.maybe_shard(logits, ap_out, self.mesh_axis_names)

    # Soft cap logits if applicable
    if self.soft_cap_logits:
      logits = self.soft_cap_logits * jnp.tanh(logits / self.soft_cap_logits)

    # abs cap logits if applicable
    if self.logits_abs_max:
      logits = jnp.clip(logits, -self.logits_abs_max, self.logits_abs_max)
    return logits

  def __call__(
      self,
      inputs: JTensor,
      class_weights: JTensor,
      class_ids: Optional[JTensor] = None,
      class_probabilities: Optional[JTensor] = None,
  ) -> NestedMap:
    """Computes logits, cross entropy etc.

    Args:
      inputs: a single JTensor with shape [..., input_dim].
      class_weights: a JTensor with shape [..., 1] containing the weights for
        each target word.
      class_ids: a JTensor with shape [..., 1] of int32 dtype containing the
        target class labels.
      class_probabilities: a JTensor with shape [..., num_classes] of float
        values indicating class-membership probabilities.

    Returns:
      A `.NestedMap` containing the following fields

      - logits: with shape [..., num_classes]. Unnormalized softmax's logits.
      - per_example_argmax: with shape [...]. argmax of i-th example.
      - per_example_xent: with shape [...]. Cross entropy between i-th example's
        prediction and its label.
      - per_example_weight: with shape [...]. class_weights casted to
        this layer's dtype.
      - total_xent: A scalar. The sum of per_example_weight * per_example_xent.
      - total_weight: A scalar. The sum of per_example_weight.
      - avg_xent: A scalar. total_loss / total_weight.
    """
    # Assert one of class_ids or class_probabilities is not None
    if class_ids is None and class_probabilities is None:
      raise ValueError('One of class_ids or class_probabilities must be given.')

    # Compute logits
    inputs_dtype = inputs.dtype
    logits = self.get_logits(inputs)
    # We perform softmax in float32 to improve stability.
    logits = logits.astype(jnp.float32)
    log_probs = jax.nn.log_softmax(logits)

    if class_probabilities is None:
      class_probabilities = jax.nn.one_hot(
          jnp.squeeze(class_ids, axis=-1), self.num_classes, dtype=jnp.float32
      )

    class_probabilities_prior_to_label_smoothing = None
    if self.label_smoothing_prob > 0.0 and not self.do_eval:
      class_probabilities_prior_to_label_smoothing = class_probabilities
      off_value = self.label_smoothing_prob / self.num_classes
      on_value = 1.0 - self.label_smoothing_prob + off_value
      class_probabilities = (
          on_value * class_probabilities
          + off_value * (1.0 - class_probabilities)
      ).astype(jnp.float32)
    class_probabilities = jax.lax.stop_gradient(class_probabilities)

    per_example_xent = -jnp.sum(
        log_probs * class_probabilities, axis=-1, dtype=jnp.float32
    )

    per_example_argmax = jax.lax.stop_gradient(
        jnp.argmax(logits.astype(jnp.float32), axis=-1)
    )

    # Compute total softmax for the entire sequence
    total_xent = jnp.sum(
        jnp.expand_dims(per_example_xent, axis=-1) * class_weights,
        dtype=jnp.float32,
    )

    total_weight = jnp.sum(class_weights, dtype=jnp.float32)

    if self.use_tgt_labels_size_as_loss_denominator:
      loss_denominator = jnp.sum(
          jnp.ones_like(class_weights), dtype=jnp.float32
      )
    else:
      loss_denominator = total_weight
    avg_xent = (total_xent / loss_denominator).astype(jnp.float32)
    z_loss = jnp.sum(_compute_z_loss(logits) * class_weights) / loss_denominator
    z_loss *= self.z_loss_weight
    self.add_summary('aux_z_loss', z_loss)
    self.add_aux_loss('aux_z_loss', z_loss)

    if class_probabilities_prior_to_label_smoothing is not None:
      per_example_xent_prior_to_label_smoothing = -jnp.sum(
          log_probs * class_probabilities_prior_to_label_smoothing,
          axis=-1,
          dtype=jnp.float32,
      )
      total_xent_prior_to_label_smoothing = jnp.sum(
          jnp.expand_dims(per_example_xent_prior_to_label_smoothing, axis=-1)
          * class_weights,
          dtype=jnp.float32,
      )
      avg_xent_prior_to_label_smoothing = (
          total_xent_prior_to_label_smoothing / loss_denominator
      ).astype(inputs_dtype)
      self.add_summary(
          'avg_xent_prior_to_label_smoothing', avg_xent_prior_to_label_smoothing
      )

    output_nmap = NestedMap(
        logits=logits.astype(inputs_dtype),
        log_probs=log_probs.astype(inputs_dtype),
        per_example_argmax=per_example_argmax,
        per_example_xent=per_example_xent.astype(jnp.float32),
        total_xent=total_xent.astype(inputs_dtype),
        # base_model.py _compute_xent_loss_helper uses avg_xent_weight if set,
        # this helper is currently used by LanguageModel only, if we have
        # EncoderDecoder model we will have to adjust weighting as well.
        avg_xent_weight=loss_denominator,
        avg_xent=avg_xent,
        total_weight=total_weight,
    )

    return output_nmap

  def extend_step(self, ids: JTensor, *, time_step: JTensor) -> JTensor:
    del time_step  # Not used.
    return self.emb_lookup(ids)


class PositionalEmbedding(base_layer.BaseLayer):
  """Generates position embedding for a given 1-d sequence.

  Attributes:
    min_timescale: Start of the geometric index. Determines the periodicity of
      the added signal.
    max_timescale: End of the geometric index. Determines the frequency of the
      added signal.
    embedding_dims: Dimension of the embedding to be generated.
  """

  min_timescale: int = 1
  max_timescale: int = 10_000
  embedding_dims: int = 0

  def __call__(
      self, seq_length: Optional[int] = None, position: Optional[JTensor] = None
  ) -> JTensor:
    """Generates a JTensor of sinusoids with different frequencies.

    Args:
      seq_length: an optional Python int definiing the output sequence length.
        if the `position` argument is specified.
      position:   [B, seq_length], optional position for each token in the
        sequence, only required when the sequence is packed.

    Returns:
      [B, seqlen, D] if `position` is specified, else [1, seqlen, D]
    """
    if position is None:
      assert seq_length is not None
      # [1, seqlen]
      position = jnp.arange(seq_length, dtype=jnp.float32)[jnp.newaxis, :]
    else:
      assert position.ndim == 2, position.shape

    num_timescales = self.embedding_dims // 2
    log_timescale_increment = math.log(
        float(self.max_timescale) / float(self.min_timescale)
    ) / jnp.maximum(jnp.asarray(num_timescales, dtype=jnp.float32) - 1, 1)
    inv_timescales = self.min_timescale * jnp.exp(
        jnp.arange(num_timescales, dtype=jnp.float32) * -log_timescale_increment
    )
    scaled_time = (
        position[:, :, jnp.newaxis]
        * inv_timescales[jnp.newaxis, jnp.newaxis, :]
    )
    signal = jnp.concatenate(
        [jnp.sin(scaled_time), jnp.cos(scaled_time)], axis=2
    ).astype(self.fprop_dtype)
    # Force usage of `np` rather than `jnp` to compute static values at trace
    # time.
    signal = jnp.pad(
        signal, [[0, 0], [0, 0], [0, np.mod(self.embedding_dims, 2)]]
    )
    return signal


class PositionalEmbedding2D(base_layer.BaseLayer):
  """Generates 2-d position embedding for sequence of flattened patches.

  See description in the ViT paper section D4. The only difference is that we
  allow the number of patches along the h and w dimensions to be different.
  https://arxiv.org/pdf/2010.11929v2.pdf

  Attributes:
    h: An integer as fixed length of image height.
    w: An integer as fixed length of image width.
    embedding_dims: An integer as dimension of embedding.
    pos_transform: Indicate how to merge the h and w.
    num_prepend_cls_tokens: Number of prepended CLS tokens.
    num_append_cls_tokens: Number of appended CLS tokens.
  """

  h: int = 0
  w: int = 0
  embedding_dims: int = 0
  pos_transform: str = 'hwd->(hw)d'
  num_prepend_cls_tokens: int = 0
  num_append_cls_tokens: int = 0

  def _compute_1d_embeddings(
      self, position: JTensor, hidden_dim: int, dtype: jnp.dtype = jnp.float32
  ):
    position = position.astype(dtype)
    half_hid = hidden_dim // 2
    freq_seq = jnp.arange(half_hid, dtype=dtype)
    # the base 10000 is from the original sinusoidal positional embedding
    # formulation introduced in "attention is all you need" section 3.5.
    # https://arxiv.org/pdf/1706.03762.pdf
    inv_freq = 1 / (10000 ** (freq_seq / half_hid))
    positions = jnp.einsum('S,D->SD', position, inv_freq)
    sin = jnp.sin(positions)
    cos = jnp.cos(positions)
    return jnp.concatenate([sin, cos], axis=-1)

  def _compute_2d_embeddings(self):
    dim = self.embedding_dims
    h_seq = jnp.arange(-self.h / 2, self.h / 2)
    w_seq = jnp.arange(-self.w / 2, self.w / 2)
    pos_emb_h = self._compute_1d_embeddings(h_seq, dim // 2, dtype=jnp.float32)
    pos_emb_w = self._compute_1d_embeddings(w_seq, dim // 2, dtype=jnp.float32)
    pos_emb_2d = jnp.concatenate(
        [
            jnp.tile(pos_emb_h[:, None, :], [1, self.w, 1]),
            jnp.tile(pos_emb_w[None, :, :], [self.h, 1, 1]),
        ],
        axis=-1,
    )
    return pos_emb_2d

  def __call__(
      self, seq_length: Optional[int] = None, position: Optional[JTensor] = None
  ) -> JTensor:
    """Generates a JTensor of sinusoids with different frequencies.

    Args:
      seq_length: Only to follow signature of PositionalEmbedding layer.
      position: Only to follow signature of PositionalEmbedding layer.

    Returns:
      2D positional embedding Tensor of shape
        [1, p.num_prepend_cls_tokens + p.h * p.w + p.num_append_cls_tokens, D].
    """
    del seq_length, position
    pos_emb = self._compute_2d_embeddings()
    pos_emb = jnp.reshape(pos_emb, (self.h * self.w, self.embedding_dims))
    if self.num_prepend_cls_tokens:
      pos_emb = jnp.concatenate(
          [
              jnp.zeros([self.num_prepend_cls_tokens, self.embedding_dims]),
              pos_emb,
          ],
          axis=0,
      )
    if self.num_append_cls_tokens:
      pos_emb = jnp.concatenate(
          [
              pos_emb,
              jnp.zeros([self.num_append_cls_tokens, self.embedding_dims]),
          ],
          axis=0,
      )
    pos_emb = jnp.expand_dims(pos_emb, axis=0)
    return pos_emb

def precompute_freqs_cis(dim: int, end: int, theta: float=10000.0, dtype: jnp.dtype=jnp.float32) -> jnp.ndarray: # XD: from EasyLM
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2)[: (dim // 2)].astype(dtype) / dim))
    t = np.arange(end)  # type: ignore
    freqs = np.outer(t, freqs).astype(dtype)  # type: ignore
    sin, cos = np.sin(freqs), np.cos(freqs)
    freqs_cis = np.complex64(cos + 1j * sin)
    return jnp.asarray(freqs_cis)

def apply_rotary_emb(  # XD: from EasyLM
    xq: jnp.ndarray,
    # xk: jnp.ndarray,
    freqs_cis: jnp.ndarray,
    dtype: jnp.dtype=jnp.float32,
):

    reshape_xq = xq.astype(jnp.float32).reshape(*xq.shape[:-1], -1, 2)
    # reshape_xk = xk.astype(jnp.float32).reshape(*xk.shape[:-1], -1, 2)

    xq_ = jax.lax.complex(reshape_xq[..., 0], reshape_xq[..., 1])
    # xk_ = jax.lax.complex(reshape_xk[..., 0], reshape_xk[..., 1])

    # add head dim
    # freqs_cis = jnp.reshape(freqs_cis, (*freqs_cis.shape[:2], 1, *freqs_cis.shape[2:]))
    freqs_cis = jnp.reshape(freqs_cis, (*freqs_cis.shape[:1], 1, *freqs_cis.shape[1:]))  # XD

    xq_out = xq_ * freqs_cis
    xq_out = jnp.stack((jnp.real(xq_out), jnp.imag(xq_out)), axis=-1).reshape(*xq_out.shape[:-1], -1)

    # xk_out = xk_ * freqs_cis
    # xk_out = jnp.stack((jnp.real(xk_out), jnp.imag(xk_out)), axis=-1).reshape(*xk_out.shape[:-1], -1)

    return xq_out.astype(dtype)#, xk_out.astype(dtype)

class RotaryPositionalEmbedding(PositionalEmbedding):
  """Applies rotary position embedding for a given 1-d sequence.

  The Rotary position embedding is described in https://arxiv.org/abs/2104.09864

  Attributes:
    cast_as_fprop_dtype: If True, the returned vars are cast as fprop_dtype to
      save some memory.
  """

  cast_as_fprop_dtype: bool = True

  def setup(self) -> None:
    if self.embedding_dims % 2:
      raise ValueError(
          'Embedding dim for rotary position embedding must be a multiple of 2.'
      )
    self.freqs_cis = precompute_freqs_cis(self.embedding_dims, 2048)  # XD
    super().setup()

  def __call__(
      self,  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
      inputs: JTensor,
      position: Optional[JTensor] = None,
  ) -> JTensor:
    """Generates a JTensor of sinusoids with different frequencies.

    Args:
      inputs: The input sequence on which to apply the Rotary position
        embedding. Since rotary position embeddings are applied to query and
        keys after projection, it is assumed of shape [B, S, N, H].
      position: Optional position JTensor which denotes the position of each
        token in the sequence. This only needs to be supplied when the sequence
        is packed. It is of shape [B, S].

    Returns:
      a JTensor of shape [B, S, N, H] which includes the inputs together with
      the rotary position embedding incorporated in it.
    """
    if len(inputs.shape) != 4:
      raise ValueError(
          'Input is assumed to be a rank 4 tensor of shape'
          '[batch, sequence, heads, dims].'
      )
    if self.embedding_dims != inputs.shape[3]:
      raise ValueError(
          'The embedding dims of the rotary position embedding'
          'must match the hidden dimension of the inputs.'
      )
    # return apply_rotary_emb(inputs, freqs_cis=self.freqs_cis, dtype=jnp.float32)#fprop_dtype  # XD
    half_embedding_dim = self.embedding_dims // 2
    fraction = 2 * jnp.arange(0, half_embedding_dim) / self.embedding_dims
    timescale = (
        self.min_timescale
        * (self.max_timescale / self.min_timescale) ** fraction
    )
    if position is None:
      seq_length = inputs.shape[1]
      position = jnp.arange(seq_length, dtype=jnp.float32)[jnp.newaxis, :]
    position = position[:, :, jnp.newaxis, jnp.newaxis]
    timescale = timescale[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]
    sinusoid_inp = position / timescale
    sin = jnp.sin(sinusoid_inp)
    cos = jnp.cos(sinusoid_inp)
    first_half, second_half = jnp.split(inputs, 2, axis=-1)
    first_part = first_half * cos - second_half * sin
    second_part = second_half * cos + first_half * sin
    # TODO(b/252874053): Clean this up after phase 3 is done.
    if self.cast_as_fprop_dtype:
      first_part = first_part.astype(self.fprop_dtype)
      second_part = second_part.astype(self.fprop_dtype)
    return jnp.concatenate([first_part, second_part], axis=-1)

  def extend_step(
      self, inputs: JTensor, position: Optional[Union[int, JTensor]] = None
  ) -> JTensor:
    """Generates a JTensor of sinusoids with different frequencies for a step.

    Args:
      inputs: The input sequence on which to apply the Rotary position
        embedding. Since rotary position embeddings are applied to query and
        keys after projection, it is assumed of shape [B, N, H] or of shape [B,
        P, N, H] where P may be a prefix length.
      position: The position which is being decoded, this should correspond to
        the logical position of the last token in the prefix window (P) in the
        entire sequence length S. It is a scalar or having shape [B].

    Returns:
      a JTensor of the same shape as input with the rotary position embedding
      incorporated in it.
    """
    assert len(inputs.shape) in [3, 4]
    inputs_shape = inputs.shape
    if len(inputs_shape) == 3:
      inputs = inputs[:, jnp.newaxis, :, :]
    seq_length = inputs.shape[1]
    # Adjust the prefix's position with position.
    # Note that position may be a tracer rather than an int, and so we must use
    # jax.lax.iota, rather than jnp.arange.
    prefix_position = jax.lax.iota(dtype=jnp.int32, size=seq_length)
    # [B, 1]
    position = jnp.broadcast_to(position, [inputs_shape[0]])[:, jnp.newaxis]
    # [B, P]
    prefix_position = position - jnp.flip(prefix_position)[jnp.newaxis, :]
    prefix_position = jnp.where(
        prefix_position < 0, jnp.zeros_like(prefix_position), prefix_position
    )
    output = self(inputs, position=prefix_position)
    if len(inputs_shape) == 3:
      output = jnp.squeeze(output, axis=1)
    return output

def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return jnp.concatenate((-x2, x1), axis=x1.ndim - 1)

def apply_rotary_pos_emb(inputs, cos, sin, offset: int = 0):
    # cos, sin = (
    #     cos[offset : inputs.shape[0] + offset, ...],
    #     sin[offset : inputs.shape[0] + offset, ...],
    # )
    return (inputs * cos) + (rotate_half(inputs) * sin)

class PythiaRotaryPositionalEmbedding(PositionalEmbedding):
    cast_as_fprop_dtype: bool = True
    rotary_pct: float = 0.25

    def setup(self) -> None:
        if self.embedding_dims % 2:
            raise ValueError("Embedding dim for rotary position embedding must be a multiple of 2.")
        super().setup()
        
    def __call__(
        self,  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
        inputs: JTensor,
        position: Optional[JTensor] = None,
        offset: int = 0,
        name: str = '',
    ) -> JTensor:
        if len(inputs.shape) != 4:
            raise ValueError(
                "Input is assumed to be a rank 4 tensor of shape[batch, sequence, heads, dims]."
            )
        if self.embedding_dims != inputs.shape[3]:
            raise ValueError(
                "The embedding dims of the rotary position embedding"
                "must match the hidden dimension of the inputs."
            )
        
        rotary_ndims = int(self.embedding_dims * self.rotary_pct)
        fraction = jnp.arange(0, rotary_ndims, 2).astype(jnp.float32) / rotary_ndims
        timescale = self.min_timescale * (self.max_timescale / self.min_timescale) ** fraction
        
        if position is None:
            seq_length = inputs.shape[1]
            # 调换了length维度顺序
            position = jnp.arange(seq_length, dtype=jnp.float32)[:, jnp.newaxis]
        else:
            if len(position.shape) == 2:
                assert position.shape[1] == inputs.shape[1]
                position = position[0]
            position = position[jnp.newaxis, :].astype(jnp.float32)

        position = position[:, :, jnp.newaxis, jnp.newaxis]
        timescale = timescale[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]
        sinusoid_inp = position / timescale
        sinusoid_inp = jnp.concatenate([sinusoid_inp, sinusoid_inp], axis=-1)
        
        if self.fprop_dtype == jnp.bfloat16:
            sinusoid_inp = sinusoid_inp.astype(jnp.float32)
            
        sin = jnp.sin(sinusoid_inp)
        cos = jnp.cos(sinusoid_inp)
        # self.add_summary("[lsp]cos", cos, verbosity=self.user_summary_level)
        # self.add_summary("[lsp]sin", sin, verbosity=self.user_summary_level)
        # self.add_summary(f"[lsp]rotary_inputs_{name}", inputs, verbosity=self.user_summary_level)
        inputs_rot, inputs_pass = (inputs[..., :rotary_ndims], inputs[..., rotary_ndims :], )
        # self.add_summary(f"[lsp]inputs_rot_{name}", inputs_rot, verbosity=self.user_summary_level)
        inputs_layer = apply_rotary_pos_emb(inputs_rot, cos, sin, offset=offset)
        inputs_layer = jnp.concatenate((inputs_layer, inputs_pass), axis=-1)

        if self.cast_as_fprop_dtype:
            inputs_layer = inputs_layer.astype(self.fprop_dtype)

        # self.add_summary(f"[lsp]rotary_inputs_layer_{name}", inputs_layer, verbosity=self.user_summary_level)
        return inputs_layer

class TrainablePositionalEmbedding(PositionalEmbedding):
  """Generates trainable position embedding for a given 1-d sequence.

  Attributes:
    max_seq_length: Max sequence length.
    lookup_style: Style of lookup, one of index or matmul.
  """

  max_seq_length: int = 10_240
  lookup_style: str = 'matmul'
  array_lookup_tpl: LayerTpl = template_field(base_ops.ArrayLookup)
  einsum_tpl: LayerTpl = template_field(base_ops.EinsumOp)

  class ActivationSharding(base_layer.BaseLayer.ActivationSharding):
    """Represents how intermediate values should be partitioned across a mesh.

    Attributes:
      emb_out_split_dims_mapping: Mesh split for embedding outputs.
    """

    emb_out_split_dims_mapping: SplitDimsMapping = None

  def setup(self) -> None:
    super().setup()
    wp = self.weight_split_dims_mapping
    self.create_variable(
        'emb_var',
        WeightHParams(
            shape=[self.max_seq_length, self.embedding_dims],
            mesh_shape=self.mesh_shape,
            tensor_split_dims_mapping=wp.wt,
        ),
    )
    self.create_child('array_lookup', self.array_lookup_tpl.clone())
    self.create_child('einsum', self.einsum_tpl.clone())

  def __call__(
      self, seq_length: Optional[int] = None, position: Optional[JTensor] = None
  ) -> JTensor:
    """Generates a JTensor of embedding lookup result.

    Args:
      seq_length: Sequence length of the embeddings to be generated. This may be
        omitted if an explicit position JTensor is specified.
      position: Optional position JTensor which denotes the position of each
        token in the sequence. This only needs to be supplied when the sequence
        is packed. It is of shape [batch, seq_length].

    Returns:
      a JTensor of shape [batch, seq_length, embedding_dim] if position JTensor
      is specified, else of shape [1, seq_length, embedding_dim].
    """
    ap = self.activation_split_dims_mapping
    if position is None:
      assert seq_length is not None
      position = jnp.arange(seq_length, dtype=jnp.float32)[jnp.newaxis, :]
    if seq_length is None:
      assert position is not None
      assert position.ndim == 2
      seq_length = position.shape[1]

    pos_emb_var = self.theta.emb_var
    pos_emb_var = jax.lax.slice_in_dim(pos_emb_var, 0, seq_length, axis=0)
    if self.lookup_style == 'index':
      embs = self.array_lookup(jnp.asarray(pos_emb_var), (position,))
    elif self.lookup_style == 'matmul':
      one_hot_ids = jax.nn.one_hot(position, seq_length, dtype=self.fprop_dtype)
      embs = self.einsum('...y,yz->...z', one_hot_ids, pos_emb_var)
    else:
      raise ValueError('Unknown lookup style.')

    embs = base_layer.maybe_shard(
        embs, ap.emb_out_split_dims_mapping, self.mesh_axis_names
    )
    return embs

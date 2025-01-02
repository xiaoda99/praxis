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

"""Transformer-related layers."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple, Union

from absl import logging
from flax import linen as nn
import jax
from jax import numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from einops import rearrange, repeat  # XD
import numpy as np
import math
from praxis import base_layer
from praxis import gshard_utils
from praxis import pax_fiddle
from praxis import py_utils
from praxis import pytypes
from praxis.layers import activations as activations_lib
from praxis.layers import attentions
from praxis.layers import checkpoint_policy
from praxis.layers import linears
from praxis.layers import normalizations
from praxis.layers import pipeline
from praxis.layers import repeats
from praxis.layers import stats
from praxis.layers import stochastics
from praxis.layers import GELU
from praxis.layers import rnn_cell
from praxis.layers.ssm import Mamba2

NestedMap = py_utils.NestedMap
WeightInit = base_layer.WeightInit
WeightHParams = base_layer.WeightHParams
template_field = base_layer.template_field
LayerTpl = pax_fiddle.Config[base_layer.BaseLayer]

JTensor = pytypes.JTensor
NestedJTensor = pytypes.NestedJTensor

SplitDimsMapping = pytypes.SplitDimsMapping
AutodiffCheckpointType = checkpoint_policy.AutodiffCheckpointType


def _rms(x):
  # Note: under pmap .mean() will produce a local mean, not across all hosts.
  return (x**2.).mean().astype(jnp.float32)**.5

def _contains_one(string, s_l):
  return any([s in string for s in s_l])


def _rel_cos(x, y):
  """Computes cosine similarity between residual x and layer output y."""
  xx = (x * x).sum(-1)
  xy = (x * y).sum(-1)
  yy = (y * y).sum(-1)
  xx += (xx == 0).astype(x.dtype)
  yy += (yy == 0).astype(y.dtype)
  xx = xx.astype(jnp.float32)
  yy = yy.astype(jnp.float32)
  xy = xy.astype(jnp.float32)
  cos_xy = xy * (xx**-0.5) * (yy**-0.5)
  return cos_xy.mean()


def compute_attention_masks_for_fprop(
    inputs: JTensor,
    paddings: Optional[JTensor] = None,
    causal_attention: Optional[bool] = False,
    segment_mask: Optional[JTensor] = None,
    cross_inputs: Optional[JTensor] = None,
    cross_paddings: Optional[JTensor] = None,
    cross_segment_mask: Optional[JTensor] = None,
    fold_padding_with_segment_mask: Optional[bool] = False,
    # window_size: Optional[Sequence[int]] = (),  # XD
) -> Tuple[JTensor, Union[JTensor, None]]:
  """Computes attention mask from paddings, segment masks etc for fprop.

  Args:
    inputs: Input sequence JTensor of shape [B, T, H].
    paddings: Input paddings JTensor of shape [B, T] (optional). Note that one
      of paddings or segment_mask must be provided.
    causal_attention: Boolean to apply causal masking (optional).
    segment_mask: Segment mask JTensor for packed input of shape [B, 1, T, T]
      ready to add to logits (optional).
    cross_inputs: Output JTensor of the encoder, to be used for cross attention,
      of shape [B, S, H].
    cross_paddings: Paddings JTensor for cross atention of shape [B, S].
    cross_segment_mask: Segment mask JTensor for encoder-decoder in packed input
      case of shape [B, 1, T, S].
    fold_padding_with_segment_mask: If True then segment mask is supposed to
      include the padding mask as well, i.e. treating PADs as one sequence and
      non-PADs as another.

  Returns:
    attention_mask: Attention mask JTensor ready to add to logits for self
      attention of shape [1|B, 1, 1|T, T].
    cross_attention_mask: Attention mask ready to add to logits for cross
      attention of shape [1|B, 1, 1|T, S]. This will be None if cross_inputs
      are None.
  """
  if fold_padding_with_segment_mask:
    # In this case a separate padding mask is not needed, it is assumed
    # folded with segment mask.
    assert segment_mask is not None
    attention_mask = segment_mask
  else:
    # Paddings must be provided to create the attention mask
    assert paddings is not None
    # Get paddings mask to [B, 1, 1, T]
    attention_mask = attentions.convert_paddings_to_mask(paddings, inputs.dtype)

    # Additional segment_mask may also be provided in this case
    if segment_mask is not None:
      attention_mask = jnp.minimum(attention_mask, segment_mask)

  # Causal mask of shape [1, 1, T, T]
  if causal_attention:
    causal_mask = attentions.causal_mask(inputs)
    attention_mask = jnp.minimum(attention_mask, causal_mask)

  # Compute cross attention mask if applicable
  cross_attention_mask = None
  if cross_inputs is not None:
    assert cross_paddings is not None

    # Compute paddings
    cross_attention_mask = attentions.convert_paddings_to_mask(
        cross_paddings, dtype=cross_inputs.dtype)

    # Packed inputs
    if cross_segment_mask is not None:
      cross_attention_mask = jnp.minimum(cross_attention_mask,
                                         cross_segment_mask)
  return attention_mask, cross_attention_mask


def compute_attention_masks_for_extend_step(
    time_step: JTensor,
    seq_len: int,
    segment_mask: Optional[JTensor] = None,
    cross_paddings: Optional[JTensor] = None,
    cross_segment_mask: Optional[JTensor] = None
) -> Tuple[JTensor, Union[JTensor, None]]:
  """Computes attention mask from paddings, segment masks etc for extend_step.

  Args:
    time_step: Time step for which to generate causal mask.
    seq_len: Sequence length for generating causal mask.
    segment_mask: if not None, per step segment mask JTensor for this time step,
      of shape [B, 1, T].
    cross_paddings: Source paddings JTensor of shape [B, S].
    cross_segment_mask: if not None, cross_segment_mask JTensor for this time
      step, of shape [B, 1, S].

  Returns:
    attention_mask: Attention mask JTensor ready to add to logits for self
      attention of shape [1|B, 1, T].
    cross_attention_mask: Attention mask JTensor ready to add to logits for
      cross attention of shape [1|B, 1, 1, S]. This will be None if
      cross_paddings are None.
  """
  # Create a broadcast friendly version of time step of shape [1, 1]
  batch_time_step = jnp.asarray(time_step, dtype=jnp.uint32)
  batch_time_step = jnp.reshape(batch_time_step, [1, 1])

  # Create causal padding by masking out any index > time_step.
  # [1, T], 0 for non-pad and 1 for pad.
  causal_padding = jnp.greater(
      jnp.expand_dims(jnp.arange(seq_len), 0), batch_time_step)

  # Create attention mask from padding of shape [1|B, 1, T]
  attention_mask = jnp.squeeze(
      attentions.convert_paddings_to_mask(causal_padding), axis=1)

  # Include segment mask, has shape [B, 1, T]
  if segment_mask is not None:
    attention_mask = jnp.minimum(attention_mask, segment_mask)

  # Compute cross attention mask if applicable
  cross_attention_mask = None
  if cross_paddings is not None:
    # Compute paddings mask [B, 1, 1, S]
    cross_attention_mask = attentions.convert_paddings_to_mask(
        cross_paddings, dtype=attention_mask.dtype)

    # Cross segment mask may be overloaded
    if cross_segment_mask is not None:
      # [B, 1, S] -> [B, 1, 1, S]
      cross_segment_mask = jnp.expand_dims(cross_segment_mask, axis=1)
      cross_attention_mask = jnp.minimum(cross_attention_mask,
                                         cross_segment_mask)
  return attention_mask, cross_attention_mask


def _get_sentence_embeddings(inputs: JTensor, segment_ids: JTensor) -> JTensor:
  """Returns the average sentence embedding to gate by.

  Args:
    inputs: Input sequence JTensor of the shape [S, D]
    segment_ids:  A JTensor of shape [S] specifying the segment each token
      belongs to.

  Returns:
    sentence_embeddings: A JTensor of the shape [S, D] that is an average of the
    input tensor per segment. The sentence embeddings will contain the same
    averaged vector for a particular segment_id.
  """
  # Max segments is required for segment_sum to be statically compiled.
  # max_segments = S
  max_segments = inputs.shape[0]
  # Compute the sum within segments(specified by segment_ids) of the input.
  # segment_sum shape: [max_segments, D]
  segment_sum = jax.ops.segment_sum(
      inputs, segment_ids, num_segments=max_segments)

  # Zero out the segment_sum belonging to segment_ids=0, which corresponds to
  # padding.
  segment_sum = segment_sum.at[0].set(0.0)

  # Compute the number of elements per segment. This is used to calculate the
  # mean.
  # num_elements_per_segment shape: [max_segments, D]
  num_elements_per_segment = jax.ops.segment_sum(
      jnp.ones_like(segment_ids), segment_ids, num_segments=max_segments)

  # segment_mean shape: [max_segments, D]
  segment_mean = segment_sum / jnp.maximum(
      num_elements_per_segment[:, jnp.newaxis], 1)
  # Sentence embedding contains the average of the input tensor per segment.
  # The sentence embeddings will contain the same averaged vector for a
  # particular segment_id.
  # sentence_embedding shape: [S, D]
  sentence_embeddings = segment_mean[segment_ids]
  return sentence_embeddings


class TransformerFeedForward(base_layer.BaseLayer):
  """Transformer feedforward layer with residual connection and dropout.

  Attributes:
    input_dims: Depth of the input.
    output_dims: Depth of the output. The value of input_dims will be used when
      output_dims is 0. Must be equal to input_dims if add_skip_connection=True.
    hidden_dims: Hidden dimension of FFN.
    has_bias: Adds bias weights to Feedforward or not.
    apply_padding_first: Apply padding to inputs before everything else or not.
      For example, it is better to apply padding before batch norm.
    activation_tpl: Activation function to use.
    use_gated_activation: Boolean indicating whether to use a gated activation
      function for the first feedforward layer or not.
    fflayer_tpl: Parameterization of the feedforward layer.
    ln_tpl: Parameterization of the layer normalization layer. Other options
      include RmsNorm as well.
    residual_dropout_prob: Residual dropout.
    relu_dropout_tpl: Parameterization of the relu dropout layer. keep_prop will
      be reset to (1.0 - relu_dropout_prob).
    relu_dropout_prob: FFN dropout.
    residual_dropout_tpl: Parameterization of the residual dropout params
      template. keep_prop will be reset to (1.0 - residual_dropout_prob).
    add_skip_connection: Whether to add residual connection.
    residual_weight: Weight of the residual connection. Output = fn(x) *
      residual_weight + x.
    residual_droppath_prob: Probability at which we drop the entire residual
      path.
    norm_policy: Policy for applying normaliztion wrt. transformations. Options
      are: (1) "pre", applied before transformation. (2) "primer_hybrid",
      applied before and after transformation. (3) "post", applied after
      transformation, (4) "post_skip", applied after the skip connection.
    internal_gshard_variance_scaling_fan_in_init: Feedforward weight init
      follows uniform distribution withbound = 1.0 / sqrt(3 / dim_0).
  """
  input_dims: int = 0
  output_dims: int = 0
  hidden_dims: int = 0
  has_bias: bool = True
  apply_padding_first: bool = False
  activation_tpl: pax_fiddle.Config[
      activations_lib.BaseActivation
  ] = template_field(activations_lib.ReLU)
  use_gated_activation: bool = False
  fflayer_tpl: LayerTpl = template_field(linears.FeedForward)
  ln_tpl: LayerTpl = template_field(normalizations.LayerNorm)
  seperate_gating_ln: bool = False # mqy
  residual_dropout_prob: float = 0.0
  relu_dropout_tpl: LayerTpl = template_field(stochastics.Dropout)
  relu_dropout_prob: float = 0.0
  residual_dropout_tpl: LayerTpl = template_field(stochastics.Dropout)
  add_skip_connection: bool = True
  residual_weight: float = 1.0
  residual_droppath_prob: float = 0.0
  norm_policy: str = 'pre'
  internal_gshard_variance_scaling_fan_in_init: bool = False
  # XD
  segment_size: int = None
  residual_cross_act_proj: bool = False
  chunk_size: int = None
  output_layer_std: float = None

  class WeightSharding(base_layer.BaseLayer.WeightSharding):
    """Represents how layer's learned parameters are partitioned across a mesh.

    Attributes:
      ffn0: Weight-split dims mapping for the first ffw network.
      ffn1: Weight-split dims mapping for the second ffw network.
    """
    ffn0: SplitDimsMapping = None
    ffn1: SplitDimsMapping = None

  class ActivationSharding(base_layer.BaseLayer.ActivationSharding):
    """Represents how intermediate values should be partitioned across a mesh.

    Attributes:
      ffn0: Activation-split dims mapping for the first ffw network.
      ffn1: Activation-split dims mapping for the second ffw network.
    """
    ffn0: SplitDimsMapping = None
    ffn1: SplitDimsMapping = None

  def setup(self) -> None:

    output_dims = self.output_dims
    if output_dims == 0:
      # Make it compatible with previous implementation
      output_dims = self.input_dims

    if self.add_skip_connection and self.input_dims != output_dims:
      raise ValueError(
          'Skip connections are only supported when input_dims == output_dims '
          f'but got {self.input_dims} != {output_dims}'
      )

    wp = self.weight_split_dims_mapping
    ap = self.activation_split_dims_mapping
    # Create Layer Norm
    if self.norm_policy == 'primer_hybrid':
      ln_p = self.ln_tpl.clone()
      ln_p.dim = self.input_dims
      self.create_child('pre_layer_norm', ln_p)
      self.create_child('post_layer_norm', ln_p)
    elif self.norm_policy in ['pre', 'post', 'post_skip']:
      ln_p = self.ln_tpl.clone()
      ln_p.name = 'fflayer_ln'
      ln_p.dim = self.input_dims
      self.create_child('layer_norm', ln_p)
      if self.seperate_gating_ln:
        ln_p = self.ln_tpl.clone()
        ln_p.name = 'fflayer_ln_gating'
        ln_p.dim = self.input_dims
        self.create_child('layer_norm_gating', ln_p)
    else:
      raise ValueError('Unrecognized norm_policy: %s' % self.norm_policy)

    self._is_ffn1_gated = self.use_gated_activation
    if self._is_ffn1_gated:
      activation = pax_fiddle.Config(activations_lib.Identity)
      gate_activation = self.activation_tpl.clone()
    else:
      activation = self.activation_tpl.clone()
      gate_activation = None

    # XD
    hidden_dims = self.hidden_dims
    if self.chunk_size is not None:
      assert self.hidden_dims % self.chunk_size == 0, f'{self.hidden_dims} % {self.chunk_size} != 0'
      self.n_chunks = self.hidden_dims // self.chunk_size
      hidden_dims = self.chunk_size

    # Create the first Feedforward layer mapping to hidden dims
    ffn1_p = self.fflayer_tpl.clone()
    ffn1_p.name = 'ffn_layer1'
    ffn1_p.input_dims = self.input_dims
    ffn1_p.has_bias = self.has_bias
    ffn1_p.activation_tpl = activation
    ffn1_p.output_dims = hidden_dims  # XD
    ffn1_p.weight_split_dims_mapping.wt = wp.ffn0
    ffn1_p.activation_split_dims_mapping.out = ap.ffn0
    if self.internal_gshard_variance_scaling_fan_in_init:
      scale = (1.0 / self.input_dims) ** 0.5 * (3.0**0.5)
      ffn1_p.linear_tpl.params_init = WeightInit.Uniform(scale)
    if self.chunk_size is None: self.create_child('ffn_layer1', ffn1_p)
    else: self.create_children('ffn_layer1', [ffn1_p.clone() for _ in range(self.n_chunks)]) # XD
    if self._is_ffn1_gated:
      # This is a gated ffw network, corresponding to gshard_builder's wi0
      gate_p = self.fflayer_tpl.clone()
      gate_p.name = 'ffn_layer1_gate'
      gate_p.input_dims = self.input_dims
      gate_p.has_bias = self.has_bias
      gate_p.activation_tpl = gate_activation
      gate_p.output_dims = hidden_dims  # XD
      gate_p.weight_split_dims_mapping.wt = wp.ffn0
      gate_p.activation_split_dims_mapping.out = ap.ffn0
      if self.internal_gshard_variance_scaling_fan_in_init:
        scale = (1.0 / self.input_dims) ** 0.5 * (3.0**0.5)
        gate_p.linear_tpl.params_init = WeightInit.Uniform(scale)
      if self.chunk_size is None: self.create_child('ffn_layer1_gate', gate_p)
      else: self.create_children('ffn_layer1_gate', [gate_p.clone() for _ in range(self.n_chunks)]) # XD
    if self.segment_size is not None:  # XD
      assert self.hidden_dims % self.segment_size == 0
      self.num_segments = self.hidden_dims // self.segment_size
      pc = WeightHParams(
          shape=[self.num_segments, self.segment_size, self.segment_size], 
          mesh_shape=self.mesh_shape, tensor_split_dims_mapping=wp.ffn0[1:] + [None, None],  # == ['mdl', None, None]. wp.ffn0 = ['data', 'mdl']
          init=WeightInit.Gaussian(self.segment_size**-0.5 * (0.01 if self.residual_cross_act_proj else 1.)),
      )
      self.create_variable('cross_act_proj', pc)


    # Create RELU dropout layer
    relu_dropout_p = self.relu_dropout_tpl.clone()
    relu_dropout_p.keep_prob = 1.0 - self.relu_dropout_prob
    self.create_child('relu_dropout', relu_dropout_p)

    # Create the second Feedforward layer mapping to input dims
    ffn2_p = self.fflayer_tpl.clone()
    ffn2_p.name = 'ffn_layer2'
    ffn2_p.input_dims = hidden_dims  # XD
    ffn2_p.has_bias = self.has_bias
    ffn2_p.activation_tpl = pax_fiddle.Config(activations_lib.Identity)
    ffn2_p.output_dims = output_dims
    ffn2_p.weight_split_dims_mapping.wt = wp.ffn1
    ffn2_p.activation_split_dims_mapping.out = ap.ffn1
    if self.internal_gshard_variance_scaling_fan_in_init:
      scale = (1.0 / hidden_dims) ** 0.5 * (3.0**0.5)  # XD
      ffn2_p.linear_tpl.params_init = WeightInit.Uniform(scale)
    elif self.output_layer_std is not None:  # XD
      ffn2_p.linear_tpl.params_init = WeightInit.Gaussian(self.output_layer_std)
    if self.chunk_size is None: self.create_child('ffn_layer2', ffn2_p)
    else: self.create_children('ffn_layer2', [ffn2_p.clone() for _ in range(self.n_chunks)]) # XD

    # Create residual dropout layer
    residual_dropout_p = self.residual_dropout_tpl.clone()
    residual_dropout_p.keep_prob = 1.0 - self.residual_dropout_prob
    self.create_child('residual_dropout', residual_dropout_p)

    if self.residual_droppath_prob > 0:
      assert self.add_skip_connection
      droppath_p = pax_fiddle.Config(
          stochastics.StochasticResidual,
          name='residual_droppath',
          survival_prob=1.0 - self.residual_droppath_prob,
      )
      self.create_child('residual_droppath', droppath_p)

  def __call__(self,
               inputs: JTensor,
               paddings: Optional[JTensor] = None,
               segment_ids: Optional[JTensor] = None,
               gate_inputs: Optional[JTensor] = None) -> JTensor:
    # Expand paddings to last dim if not None to have shape [batch, time, 1]
    if paddings is not None:
      paddings = jnp.expand_dims(paddings, axis=-1)

    if self.apply_padding_first and paddings is not None:
      inputs *= (1.0 - paddings)

    self.add_summary('input_rms', _rms(inputs), verbosity=4)
    residual = inputs
    
    if self.norm_policy == 'primer_hybrid':
      inputs = self.pre_layer_norm(inputs)
    elif self.norm_policy == 'pre':
      inputs = self.layer_norm(inputs)
      if self.seperate_gating_ln and gate_inputs is not None: # qkvgm, lidx>0
        gate_inputs = self.layer_norm_gating(gate_inputs)
      else:
        gate_inputs = inputs

    if self.norm_policy in ('primer_hybrid', 'pre'):
      self.add_summary('input_norm_rms', _rms(inputs), verbosity=4)

    if self.chunk_size is None:
      # Apply first FFN layer
      if self._is_ffn1_gated:
        # theta.ffn_layer1_gate corresponds to gshard_builder's wi0
        gate_value = self.ffn_layer1_gate(gate_inputs)
        # theta.ffn_layer1 corresponds to gshard_builder's wi1
        activations = gate_value * self.ffn_layer1(inputs)
      else:
        activations = self.ffn_layer1(inputs)
        activations = checkpoint_name(activations, 'ffn1')
      if self.segment_size is not None:
        activations0 = activations
        activations = rearrange(activations, 'B T (S N) -> B T S N', N=self.segment_size)
        activations = jnp.einsum('BTSN,SNM->BTSM', activations, self.theta.cross_act_proj)
        activations = rearrange(activations, 'B T S M -> B T (S M)')  # == BTF
        if self.residual_cross_act_proj: activations = activations + activations0

      # Apply paddings if not None
      if not self.apply_padding_first and paddings is not None:
        activations *= (1.0 - paddings)

      self.add_summary('activation_rms', _rms(activations), verbosity=4)

      # Apply RELU dropout
      activations = self.relu_dropout(activations)

      # Apply second FFN layer
      outputs = self.ffn_layer2(activations)
    else:
      outputs = None
      for i in range(self.n_chunks):
        if self._is_ffn1_gated:
          gate_value = self.ffn_layer1_gate[i](gate_inputs)
          activations = gate_value * self.ffn_layer1[i](inputs)
        else:
          activations = self.ffn_layer1[i](inputs)

        if not self.apply_padding_first and paddings is not None:
          activations *= (1.0 - paddings)

        activations = self.relu_dropout(activations)
        _outputs = self.ffn_layer2[i](activations)
        outputs = _outputs if outputs is None else outputs + _outputs

    outputs = checkpoint_name(outputs, 'ffn2')

    # Apply paddings if not None
    if not self.apply_padding_first and paddings is not None:
      outputs *= (1.0 - paddings)

    self.add_summary('output_rms', _rms(outputs), verbosity=4)

    # Apply Primer normalization before dropout.
    if self.norm_policy == 'primer_hybrid':
      outputs = self.post_layer_norm(outputs)
    elif self.norm_policy == 'post':
      outputs = self.layer_norm(outputs)

    if self.norm_policy in ('primer_hybrid', 'post'):
      self.add_summary('output_norm_rms', _rms(outputs), verbosity=4)

    # Apply residual dropout
    outputs = self.residual_dropout(outputs)

    # Apply skip connection
    if self.add_skip_connection:
      if self.residual_droppath_prob:
        outputs = self.residual_droppath(residual, outputs)
      else:
        outputs = residual + outputs * self.residual_weight

    if self.norm_policy == 'post_skip':
      outputs = self.layer_norm(outputs)

    if self.input_dims == self.output_dims:
      # Cosine similarity between inputs (residual) and outputs.
      self.add_summary(
          'output_rel_cos', _rel_cos(residual, outputs), verbosity=4)

    return outputs

  def extend_step(self,
                  inputs: JTensor,
                  *,
                  time_step: JTensor) -> JTensor:
    """Fprop FFN extend step layer."""
    del time_step  # Not used.
    return self.__call__(inputs)


class TransformerFeedForwardMoe(base_layer.BaseLayer):
  """A sharded MoE Layer.

  This is a drop-in replacement of the transformer feedforward layer. It is a
  composite of the following sub-layers.

  ln_inputs = ln(inputs)
  moe_output = moe(ln_inputs)
  drop_output = dropout(moe_output)
  output = inputs + drop_output

  Attributes:
    input_dims: Dimension of the layer input.
    hidden_dims: Dimension of the hidden layer.
    apply_padding_first: Apply padding to inputs before everything else or
      not. For example, it is better to apply padding before batch norm.
    ln_tpl: Parameterization of the layer normalization layer.
    activation_tpl: Activation function to use.
    relu_dropout_tpl: Parameterization of the relu dropout layer. keep_prop
      will be reset to (1.0 - relu_dropout_prob).
    relu_dropout_prob: Probability at which we apply dropout to the hidden
      layer of feedforward network..
    residual_dropout_tpl: Parameterization of the residual dropout params
      template. keep_prop will be reset to (1.0 - residual_dropout_prob).
    residual_dropout_prob: Probability at which we apply dropout to the
      residual layers, such that, residual(x, y) = (x + dropout(y)).
    add_skip_connection: If True, add skip_connection from input to output.
    residual_weight: Weight applied on residual connection. Final output is
      residual_weight * residual_fn(x) + x. Only in effect when
      add_skip_connection is True.
    norm_policy: Policy for applying normaliztion wrt. transformations.
      Options are: (1) "pre", applied before transformation. (2)
      "primer_hybrid", applied before and after transformation. (3) "post",
      applied after transformation.
    residual_droppath_prob: Probability at which we drop the entire residual
      path.
    gating_func: Gating function type--can be one of the following options:
      'top2', based on the GShard paper: https://arxiv.org/abs/2006.16668,
      'expert_choice', based on https://arxiv.org/abs/2202.09368,
      'dense_top2': experimental gating function for decodiing. Similar to
      'top2' gating, but no capacity constrainst for each expert.
    num_experts: Total number of experts in this layer.
    num_groups: Total number of groups for dispatching. num_groups typically
      should be the same as num devices.
    min_group_size: If not None, num_groups will be adjusted so that there
      will be at least min_group_size tokens in each group.
    expert_capacity_dim: Internal. Exact expert capacity. Setting non-zero
      unadjusted_expert_capacity_factor is a preferred way.
    unadjusted_expert_capacity_factor: Expert capacity factor. This is the
      ratio between global batch size and total capacity across all experts
      and all routing groups. If the global batch size is G*S (num_groups*
      group_size) or B*L(batch*length) and the total expert capacity across
      all routing groups is E*G*C (num_experts*num_groups*expert_capacity),
      then unadjusted_expert_capacity_factor == (E*G*C)/(G*S)
      unadjusted_expert_capacity_factor is set to 2 by default for top-2
      routing.
    expert_weight_shards: Shard each expert params into this many number of
      shards to reduce the size of individual weight params.
    second_expert_policy: How to pick second expert: all, sampling or random.
    internal_gshard_variance_scaling_fan_in_init: Internal. Do not use. To
      study MoE layer init.
    explicit_fan_in_fan_out_axes: Set to True except for backward compatibility.
      Current Transformer implementation typically does weight stacking and
      reshapes to improve efficiency, to preserve correct init scale one needs
      to specify fan_in and fan_out dimensions explicitly. When such dimensions
      are explicitly specified, receptive_field multiplier is set to 1.
    moe_load_balance_loss_weight: Weight for the load balancing loss of the
      MoE layer.
    gating_logit_cap:  Cap the absolute values of MoE gating logits by tanh.
      Enabled when a positive value is specified.
    moe_gating_embedding_level: Specifies the type of MOE gating embedding
    used.
    See Section 3.1 https://arxiv.org/pdf/2110.03742.pdf.
    Options are:
      (1) "token" -> The gating function uses tokens to route.
      (2) "sentence" -> The gating function uses the sentence embeddings,
          calculated by taking the average token representation, to route.
    use_gated_activation: Boolean indicating whether to use a gated activation
      function for the input projection layer or not.
  """
  input_dims: int = 0
  hidden_dims: int = 0
  apply_padding_first: bool = False
  ln_tpl: LayerTpl = template_field(normalizations.LayerNorm)
  activation_tpl: pax_fiddle.Config[
      activations_lib.BaseActivation
  ] = template_field(activations_lib.ReLU)
  relu_dropout_tpl: LayerTpl = template_field(stochastics.Dropout)
  relu_dropout_prob: float = 0.0
  residual_dropout_tpl: LayerTpl = template_field(stochastics.Dropout)
  residual_dropout_prob: float = 0.0
  add_skip_connection: bool = True
  residual_weight: float = 1.0
  norm_policy: str = 'pre'
  residual_droppath_prob: float = 0.0
  gating_func: str = 'top2'
  num_experts: int = 0
  num_groups: int = 0
  min_group_size: Optional[int] = None
  expert_capacity_dim: int = 0
  unadjusted_expert_capacity_factor: float = 2.0
  expert_weight_shards: int = 1
  second_expert_policy: str = 'all'
  internal_gshard_variance_scaling_fan_in_init: bool = True
  explicit_fan_in_fan_out_axes: bool = False  # TODO(b/267235257) switch to True
  moe_load_balance_loss_weight: float = 1.0
  gating_logit_cap: float = 0.0
  moe_gating_embedding_level: str = 'token'
  use_gated_activation: bool = False

  # SPMD partition related params.
  # M - model_dim, for both inputs and outputs
  # E - experts dim
  # G - groups dim
  # C - experts capacity dim
  # H - hidden dim
  # S - sequence dim

  class WeightSharding(base_layer.BaseLayer.WeightSharding):
    """Represents how layer's learned parameters are partitioned across a mesh.

    Attributes:
      me: Sharding for the gating network weight, of shape [input_dims,
        num_experts].
      emh: Sharding of the first projection matrix that maps from input to
        hidden dim, of shape [num_experts, input_dims, hidden_dims].
      ehm: Sharding of the second projection matrix that maps from hidden to
        output dim, of shape [num_experts, hidden_dims, output_dims].
    """
    me: SplitDimsMapping = None
    emh: SplitDimsMapping = None
    ehm: SplitDimsMapping = None

  class ActivationSharding(base_layer.BaseLayer.ActivationSharding):
    """Represents how intermediate values should be partitioned across a mesh.

    Attributes:
      gsm: Sharding of the gsm tensors.
      gs: Sharding of the gs tensors.
      gsec: Sharding of the gsec tensors.
      egcm: Sharding of the egcm tensors.
      gecs: Sharding of the gecs tensors.
      gec: Sharding of the gec tensors.
      egch: Sharding of the egch tensors.
      gecm: Sharding of the gecm tensors.
    """
    gsm: SplitDimsMapping = None
    gs: SplitDimsMapping = None
    gsec: SplitDimsMapping = None
    gecs: SplitDimsMapping = None
    gec: SplitDimsMapping = None
    egcm: SplitDimsMapping = None
    egch: SplitDimsMapping = None
    gecm: SplitDimsMapping = None

  def setup(self) -> None:
    assert self.name
    assert self.input_dims
    assert self.hidden_dims

    assert self.unadjusted_expert_capacity_factor or self.expert_capacity_dim
    assert self.num_experts > 0
    assert self.num_groups > 0
    assert (
        self.expert_weight_shards == 1
    ), f'[Deprecated] Should be removed {self.expert_weight_shards} != 1'

    if self.norm_policy == 'primer_hybrid':
      params = self.ln_tpl.clone()
      params.dim = self.input_dims
      self.create_child('pre_layer_norm', params)
      self.create_child('post_layer_norm', params)
    elif self.norm_policy == 'pre' or self.norm_policy == 'post':
      params = self.ln_tpl.clone()
      params.name = 'layer_norm'
      params.dim = self.input_dims
      self.create_child('layer_norm', params)
    else:
      raise ValueError('Unrecognized norm_policy: %s' % self.norm_policy)

    dropout_tpl = self.residual_dropout_tpl.clone()
    dropout_tpl.keep_prob = 1.0 - self.residual_dropout_prob
    self.create_child('residual_dropout', dropout_tpl)

    dropout_tpl = self.relu_dropout_tpl.clone()
    dropout_tpl.keep_prob = 1.0 - self.relu_dropout_prob
    self.create_child('relu_dropout', dropout_tpl)

    if self.residual_droppath_prob > 0:
      assert self.add_skip_connection
      droppath_p = pax_fiddle.Config(
          stochastics.StochasticResidual,
          name='residual_droppath',
          survival_prob=1.0 - self.residual_droppath_prob,
      )
      self.create_child('residual_droppath', droppath_p)

    self.create_child('activation', self.activation_tpl.clone())

    # Assume output_dims == input_dims
    output_dims = self.input_dims

    # First create the gating network.
    wp = self.weight_split_dims_mapping
    gate_init = None  # default xavier init
    if self.internal_gshard_variance_scaling_fan_in_init:
      # TODO(lepikhin): this init is related with Adafactor settings, study
      stddev = (1.0 / self.input_dims) ** 0.5
      gate_scale = stddev * 3.0**0.5
      gate_init = WeightInit.Uniform(gate_scale)
    gate_pc = WeightHParams(
        shape=[self.input_dims, self.num_experts],
        init=gate_init,
        mesh_shape=self.mesh_shape,
        tensor_split_dims_mapping=wp.me,
    )
    logging.debug('moe gate WeightHParams %s', gate_pc)
    self.create_variable('gate', gate_pc)

    # Next create the expert network.
    # Params initialization follows gshard_builder.py.
    # emh tensor typically mesh-shard on first dim and last dim. Hence, here we
    # split the tensor manually into multiple tensors on the second dim.
    emh_shape = [
        self.num_experts,
        self.input_dims,  # expert_weight_shards == 1
        self.hidden_dims,
    ]
    self._is_ffn1_gated = self.use_gated_activation
    wi_init = None
    if self.internal_gshard_variance_scaling_fan_in_init:
      stddev = (1.0 / self.input_dims) ** 0.5
      wi_init_scale = stddev * 3.0**0.5
      wi_init = WeightInit.Uniform(wi_init_scale)
    wi_pc = WeightHParams(
        shape=emh_shape,
        init=wi_init,
        mesh_shape=self.mesh_shape,
        tensor_split_dims_mapping=wp.emh,
        fan_in_axes=([-2] if self.explicit_fan_in_fan_out_axes else None),
        fan_out_axes=([-1] if self.explicit_fan_in_fan_out_axes else None),
    )
    logging.debug('moe wi WeightHParams %s', wi_pc)
    if self._is_ffn1_gated:
      self.create_variable('wi_gate_0', wi_pc)
    self.create_variable('wi_0', wi_pc)

    # EHM Tensor (output transformation after RELU)
    # ehm tensor typically shard on the first dim and the second dim. Here we
    # manually split the tensor on the last dim into multiple tensors.
    ehm_shape = [
        self.num_experts,
        self.hidden_dims,
        output_dims,  # expert_weight_shards == 1
    ]
    wo_init = None
    if self.internal_gshard_variance_scaling_fan_in_init:
      wi_init = None
      stddev = (1.0 / self.hidden_dims) ** 0.5
      wo_init_scale = stddev * 3.0**0.5
      wo_init = WeightInit.Uniform(wo_init_scale)
    wo_pc = WeightHParams(
        shape=ehm_shape,
        init=wo_init,
        mesh_shape=self.mesh_shape,
        tensor_split_dims_mapping=wp.ehm,
        fan_in_axes=([-2] if self.explicit_fan_in_fan_out_axes else None),
        fan_out_axes=([-1] if self.explicit_fan_in_fan_out_axes else None),
    )
    logging.debug('moe wo WeightHParams %s', wo_pc)
    self.create_variable('wo_0', wo_pc)

  def _split(self, t_in, sharding):
    return base_layer.maybe_shard(t_in, sharding, self.mesh_axis_names)

  def _get_weights(self):
    """Get the expert weights."""
    return self.theta['wi_0'], self.theta['wo_0']

  def _count_dead_neurons(self, hidden, dispatch_tensor):
    threshold = 0
    activation_class_name = self.activation_tpl.cls.__name__
    if isinstance(self.activation_tpl.cls, activations_lib.GELU):
      logging.info(
          'Setting dead neuron count threshold=-3.0 '
          'for approximate GeLU activation'
      )
      threshold = -3.0

    nonpadding_indicator = jnp.einsum('gsec->ec', dispatch_tensor)
    nonpadding_indicator = nonpadding_indicator[:, jnp.newaxis, :, jnp.newaxis]
    padding_indicator = 1 - nonpadding_indicator
    hidden_minus_ten_padding_indicator = hidden - 10 * padding_indicator
    # EG, taking max over G and C dim
    max_hidden = jnp.max(
        jnp.max(hidden_minus_ten_padding_indicator, axis=1), axis=1
    )
    dead_neuron_indicator = jnp.less(max_hidden, threshold).astype(jnp.int32)
    dead_neuron_count = jnp.count_nonzero(dead_neuron_indicator)
    self.add_summary('dead_%s_count' % activation_class_name, dead_neuron_count)

  def _combine_top2_expert_outputs(self, inputs, paddings, segment_ids):
    """Combine outputs from top 2 experts directly."""
    fprop_dtype = self.fprop_dtype
    ap = self.activation_split_dims_mapping

    theta_wi, theta_wo = self.theta['wi_0'], self.theta['wo_0']
    assert (
        not self._is_ffn1_gated
    ), 'dense_top2 routing does not support gated MoE activations'
    if self.moe_gating_embedding_level == 'sentence':
      if segment_ids is None and paddings is None:
        sentence_embeddings = jnp.tile(
            jnp.average(inputs, axis=1, keepdims=True), [1, inputs.shape[1], 1])
      else:
        if segment_ids is None:
          segment_ids = jnp.asarray(1 - paddings, jnp.int32)
        sentence_embeddings_fn = jax.vmap(_get_sentence_embeddings, (0, 0), 0)
        sentence_embeddings = sentence_embeddings_fn(inputs, segment_ids)
      logits = jnp.einsum('gsm,me->gse', sentence_embeddings, self.theta.gate)
    else:
      logits = jnp.einsum('gsm,me->gse', inputs, self.theta.gate)
    # gsm -> gse tensor
    raw_gates = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
    if raw_gates.dtype != fprop_dtype:
      raw_gates = raw_gates.astype(fprop_dtype)

    # When picking top-1 or top-2
    raw_gates_max = jnp.max(raw_gates, axis=-1, keepdims=True)
    raw_gates_thr = jnp.max(
        raw_gates * jnp.less(raw_gates, raw_gates_max).astype(fprop_dtype),
        axis=-1,
        keepdims=True)

    gates = raw_gates * jnp.greater_equal(raw_gates,
                                          raw_gates_thr).astype(fprop_dtype)

    gates_denom = jnp.sum(gates, axis=-1, keepdims=True) + 1e-12
    gates /= gates_denom

    hidden = jnp.einsum('gsm,emh->egsh', inputs, theta_wi)
    # Activation function.
    hidden = self.activation(hidden)
    # Dropout.
    hidden = self.relu_dropout(hidden)
    # Output.
    expert_output = jnp.einsum('egsh,ehm->egsm', hidden, theta_wo)
    combined_output = jnp.einsum('egsm,gse->gsm', expert_output, gates)
    combined_output = self._split(combined_output, ap.gsm)
    aux_loss = jnp.array(0.0)
    return combined_output, aux_loss

  def _dispatch_and_combine_expert_outputs(self, inputs, paddings, segment_ids):
    """Combine expert outputs using GShard-style dispatch-combine tensors."""
    fprop_dtype = self.fprop_dtype
    ap = self.activation_split_dims_mapping
    output_dims = self.input_dims
    assert self.gating_func != 'dense_top2'

    theta_wi, theta_wo = self.theta['wi_0'], self.theta['wo_0']
    if self._is_ffn1_gated:
      theta_wi_gated = self.theta['wi_gate_0']

    token_shape = inputs.shape[:-1]
    num_tokens = np.prod(token_shape)
    m_dim = inputs.shape[-1]
    if paddings is not None:
      assert paddings.shape == token_shape

    num_groups = self.num_groups
    assert num_groups
    if (
        self.min_group_size is not None
        and num_tokens / num_groups < self.min_group_size
    ):
      num_groups = (num_tokens + self.min_group_size - 1) // self.min_group_size
      logging.info('num_groups adjusted to %s.', num_groups)
    if num_tokens % num_groups != 0:
      raise ValueError(f'The value of num_groups {num_groups} does not '
                       f'evenly divide the value of num_tokens {num_tokens}')
    g_len = num_tokens // num_groups

    reshaped_inputs = inputs.reshape([num_groups, g_len, m_dim])
    reshaped_inputs = self._split(reshaped_inputs, ap.gsm)
    if paddings is not None:
      reshaped_paddings = paddings.reshape([num_groups, g_len])
      reshaped_paddings = self._split(reshaped_paddings, ap.gs)
      reshaped_paddings = reshaped_paddings.astype(fprop_dtype)
    else:
      reshaped_paddings = None
    if self.moe_gating_embedding_level == 'sentence':
      if segment_ids is None and paddings is None:
        sentence_embeddings = jnp.tile(
            jnp.average(inputs, axis=1, keepdims=True), [1, inputs.shape[1], 1])
      else:
        if segment_ids is None:
          segment_ids = jnp.asarray(1 - paddings, jnp.int32)
        # It is important that the segment_ids have shape [B, S, D] and are not
        # reshaped to [G, B*S / G, D] when calculating the sentence embeddings.
        # The vmap below assumes that it is operating on a batch.
        sentence_embeddings_fn = jax.vmap(_get_sentence_embeddings, (0, 0), 0)
        sentence_embeddings = sentence_embeddings_fn(inputs, segment_ids)
      # Reshape the sentence embeddings, so that it corresponds to sharding
      # groups.
      reshaped_sentence_embeddings = sentence_embeddings.reshape(
          [num_groups, g_len, m_dim])
      reshaped_sentence_embeddings = self._split(reshaped_sentence_embeddings,
                                                 ap.gsm)
      logits = jnp.einsum('gsm,me->gse', reshaped_sentence_embeddings,
                          self.theta.gate)
    else:
      logits = jnp.einsum('gsm,me->gse', reshaped_inputs, self.theta.gate)

    # Here and below, we assume num devices equals num groups.
    # TODO(yonghui): Expose some of the options below through params.
    # NOTE(yonghui): The following code might break during beam search decode
    # due to much smaller group size.
    # TODO(yonghui): Avoid explicitly casting everything to fp32 once
    # top2_gating_on_logits is stable in low-precision mode.
    # TODO(lepikhin): Validate stability. mask_dtype=np.int32 and
    # logits.astype(np.float32) should generally be sufficient.
    if self.second_expert_policy == 'all':
      prng_key = None
    else:
      prng_key = self.next_prng_key()
    gating = gshard_utils.compute_gating(
        paddings=reshaped_paddings,
        logits=logits.astype(jnp.float32),
        experts_dim=self.num_experts,
        expert_capacity_dim=self.expert_capacity_dim,
        fprop_dtype=fprop_dtype,
        gating_func=self.gating_func,
        prng_key=prng_key,
        second_expert_policy=self.second_expert_policy,
        second_expert_threshold=0.0,
        legacy_mtf_behavior=True,
        capacity_factor=self.unadjusted_expert_capacity_factor,
        mask_dtype=jnp.int32,
        gating_logit_cap=self.gating_logit_cap,
    )

    if self.gating_func == 'top2':
      aux_loss, combine_tensor, dispatch_tensor, summary = gating
      over_capacity_1_ratio, over_capacity_2_ratio = summary
      self.add_summary('over_capacity_1_ratio', over_capacity_1_ratio)
      self.add_summary('over_capacity_2_ratio', over_capacity_2_ratio)
    else:
      aux_loss, combine_tensor, dispatch_tensor = gating
    if fprop_dtype != np.float32:
      combine_tensor = combine_tensor.astype(fprop_dtype)
      dispatch_tensor = dispatch_tensor.astype(fprop_dtype)

    # both tensors have shape [g, s, e, c]
    if self.gating_func in ['top2', 'expert_choice_v2']:
      combine_tensor = self._split(combine_tensor, ap.gsec)
      dispatch_tensor = self._split(dispatch_tensor, ap.gsec)
      expert_inputs = jnp.einsum('gsec,gsm->egcm', dispatch_tensor,
                                 reshaped_inputs)
    elif self.gating_func == 'expert_choice':
      combine_tensor = self._split(combine_tensor, ap.gec)
      dispatch_tensor = self._split(dispatch_tensor, ap.gecs)
      expert_inputs = jnp.einsum('gecs,gsm->egcm', dispatch_tensor,
                                 reshaped_inputs)
    else:
      raise ValueError('Unsupported gating function: %s ' % self.gating_func)
    expert_inputs = self._split(expert_inputs, ap.egcm)

    if self._is_ffn1_gated:
      hidden0 = jnp.einsum('egcm,emh->egch', expert_inputs, theta_wi)
      hidden1 = jnp.einsum('egcm,emh->egch', expert_inputs, theta_wi_gated)
      if self.gating_func in ['top2', 'expert_choice_v2']:
        self._count_dead_neurons(hidden1, dispatch_tensor)
      hidden1 = self.activation(hidden1)
      hidden = hidden1 * hidden0
    else:
      hidden = jnp.einsum('egcm,emh->egch', expert_inputs, theta_wi)
      hidden = self._split(hidden, ap.egch)
      if self.gating_func in ['top2', 'expert_choice_v2']:
        self._count_dead_neurons(hidden, dispatch_tensor)
      hidden = self.activation(hidden)

    # Dropout.
    hidden = self.relu_dropout(hidden)
    # Output.
    expert_output = jnp.einsum('egch,ehm->egcm', hidden, theta_wo)
    expert_output = self._split(expert_output, ap.egcm)
    # Now transpose and reshard.
    transposed_expert_output = jnp.einsum('egcm->gecm', expert_output)
    transposed_expert_output = self._split(transposed_expert_output, ap.gecm)
    if self.gating_func in ['top2', 'expert_choice_v2']:
      combined_output = jnp.einsum('gecm,gsec->gsm', transposed_expert_output,
                                   combine_tensor)
    elif self.gating_func == 'expert_choice':
      combined_output = jnp.einsum('gecm,gecs,gec->gsm',
                                   transposed_expert_output, dispatch_tensor,
                                   combine_tensor)
    else:
      raise ValueError('Unsupported gating function: %s ' % self.gating_func)
    combined_output = self._split(combined_output, ap.gsm)

    combined_output = combined_output.reshape(token_shape + (output_dims,))
    return combined_output, aux_loss

  def __call__(self,  # pytype: disable=annotation-type-mismatch  # jax-ndarray
               inputs: JTensor,
               paddings: JTensor = None,
               segment_ids: JTensor = None) -> JTensor:
    """Layer-norm, route, feed-forward, combine, residual.

    Args:
      inputs: [batch, seq_len, model].
      paddings: [batch, seq_len], optional when called by extend_step.
      segment_ids: [batch, seq_len] Optional. Segment_ids is used when
        moe_gating_embedding_level == 'sentence'.

    Returns:
      Tensor of the same shape as inputs.
    """
    # Assume output_dims == input_dims
    fprop_dtype = self.fprop_dtype

    # Consistent with gshard implementation.
    if self.apply_padding_first and paddings is not None:
      inputs *= (1.0 - jnp.expand_dims(paddings, axis=-1))

    self.add_summary('input_rms', _rms(inputs), verbosity=4)
    residual = inputs

    # TODO(zhangqiaorjc): Handle input of shape [batch, seq_len, g, model/g]?
    if self.norm_policy == 'primer_hybrid':
      inputs = self.pre_layer_norm(inputs)
    elif self.norm_policy == 'pre':
      inputs = self.layer_norm(inputs)

    if self.norm_policy in ('primer_hybrid', 'pre'):
      self.add_summary('input_norm_rms', _rms(inputs), verbosity=4)

    assert len(inputs.shape) in [2, 3]

    if self.gating_func == 'dense_top2':
      outputs, aux_loss = self._combine_top2_expert_outputs(
          inputs, paddings, segment_ids)
    else:
      outputs, aux_loss = self._dispatch_and_combine_expert_outputs(
          inputs, paddings, segment_ids)

    # Apply padding.
    if paddings is not None:
      outputs *= (1.0 - jnp.expand_dims(paddings, -1)).astype(fprop_dtype)

    self.add_summary('output_rms', _rms(outputs), verbosity=4)

    # Primer normalization before dropout.
    if self.norm_policy == 'primer_hybrid':
      outputs = self.post_layer_norm(outputs)
    elif self.norm_policy == 'post':
      outputs = self.layer_norm(outputs)

    if self.norm_policy in ('primer_hybrid', 'post'):
      self.add_summary('output_norm_rms', _rms(outputs), verbosity=4)

    # Residual dropout.
    outputs = self.residual_dropout(outputs)
    if self.add_skip_connection:
      if self.residual_droppath_prob:
        outputs = self.residual_droppath(residual, outputs)
      else:
        outputs = residual + outputs * self.residual_weight

    self.add_summary('output_rel_cos', _rel_cos(residual, outputs), verbosity=4)

    # Add loss to a global collection. We don't return the loss to the caller
    # to avoid the change of the api here.
    assert self.moe_load_balance_loss_weight, (
        'p.moe_load_balance_loss_weight > 0 when there is an aux '
        'load balancing loss in MoE layers.'
    )
    aux_loss = aux_loss.astype(fprop_dtype)
    aux_loss *= self.moe_load_balance_loss_weight
    self.add_summary('aux_moe_load_balance_loss', aux_loss)
    self.add_aux_loss('aux_moe_load_balance_loss', aux_loss)

    return outputs

  def extend_step(self,
                  inputs: JTensor,
                  *,
                  time_step: JTensor) -> JTensor:
    """Fprop FFN extend step layer."""
    del time_step  # Not used.
    return self.__call__(inputs)


class Transformer(base_layer.BaseLayer):
  """Transformer layer with multi-headed attention.

  Attributes:
    input_dims: Dimension of the transformer block input.
    hidden_dims: Hidden dimension of FFN layer.
    num_heads: Number of heads in self-attention.
    dim_per_head: Dimension of each attention head. If None then dim_per_head ==
      hidden_dim // num_heads.
    dropout_tpl: Residual dropout params template. keep_prop will be reset to
      (1.0 - residual_dropout_prob).
    atten_dropout_prob: Probability at which we apply dropout to the attention
      weights.
    residual_dropout_prob: Probability at which we apply dropout to the residual
      layers, such that, residual(x, y) = (x + dropout(y)).
    relu_dropout_prob: Probability at which we apply dropout to the FFN layers.
    residual_droppath_prob: Probability at which we drop the entire residual
      path.
    cross_attention: If True, perform cross encoder-decoder attention.
    allow_skip_cross_attention: If True, allow skipping cross attention during
      forward pass and decoding. This allows to skip cross attention when cross
      inputs are not available. For example, if we want to train the model with
      paired data and unimodal data. For paired data, we need cross attention
      but for unimodal data, we don't have cross inputs.
    cross_atten_tpl: Optional cross attention params template that can be set
      when cross attention is enabled. If cross-attention is enabled and this is
      set to None, then cross-attention params will be inherited from
      tr_atten_tpl.
    ln_tpl: Parameterization of the layer normalization layer.
    norm_policy: Policy for applying normaliztion wrt. transformations. Options
      are: (1) "pre", applied before transformation. (2) "primer_hybrid",
      applied before and after transformation. (3) "post", applied after
      transformation. (4) "post_skip", applied after the skip connection.
    tr_atten_tpl: Parameterization of the DotProductAttention layer.
    packed_input: If True, each training example may pack multiple sequences.
    tr_fflayer_tpl: Parameterization of the transformer Feed-Forward Layer.
    ngrammer_tpl: Params for the Ngrammer layer. This param must correspond to
      the VQNgrammer layer. If this is None, then there is no NGrammer layer
      present in this layer.
  """
  input_dims: int = 0
  hidden_dims: int = 0
  num_heads: Optional[int] = None
  dim_per_head: Optional[int] = None
  dropout_tpl: LayerTpl = template_field(stochastics.Dropout)
  atten_dropout_prob: float = 0.0
  residual_dropout_prob: float = 0.0
  relu_dropout_prob: float = 0.0
  residual_droppath_prob: float = 0.0
  use_cross_attention: bool = False
  allow_skip_cross_attention: bool = False
  cross_atten_tpl: Optional[LayerTpl] = template_field(None)
  ln_tpl: LayerTpl = template_field(normalizations.LayerNorm)
  norm_policy: str = 'pre'
  tr_atten_tpl: LayerTpl = template_field(attentions.DotProductAttention)
  packed_input: bool = False
  tr_fflayer_tpl: LayerTpl = template_field(TransformerFeedForward)
  gpt_j_residual: bool = False  # XD
  ngrammer_tpl: Optional[LayerTpl] = template_field(None)
  layer_index: int = 0 
  num_layers: int = 1
  num_ffn: int = 1 # mqy; group of feedforward layer
  hyper_conn: bool = False 
  hyper_conn_n: int = 4
  hyper_conn_merge_wcdc: bool = False # whether to merge width connection and depth connection
  hyper_conn_tanh: bool = False
  hyper_conn_norm_tpl: LayerTpl = template_field(normalizations.RmsNorm)  # mqy
  dense_conn: bool = False
  dense_conn_on_attn: bool = False
  dense_conn_on_layerdiff: bool = False
  dynamic_dense: bool = False
  dynamic_dense_sep_qkv_ln: bool = False
  dynamic_dense_stack: bool = True
  dynamic_dense_type: Optional[str] = None # ['qkv', 'qkvm']
  dynamic_dense_num_groups : int = 1
  dynamic_dense_ov: bool = False
  dynamic_dense_ov_gate: bool = False
  dynamic_dense_ov_init: str = 'gauss+gauss' # ['gauss+gauss', 'gauss+const0']
  dynamic_dense_ov_rank: int = 64
  dynamic_dense_ov_outer_loop: bool = False
  dynamic_dense_hidden_expand: int = 1
  dynamic_dense_hidden_round: bool = False
  dynamic_dense_key_wise: bool = False 
  dynamic_dense_query_wise: bool = True
  dynamic_dense_multilayer: bool = True
  dynamic_dense_gate_mlp: bool = False
  dynamic_dense_glu_mlp: bool = False 
  dynamic_dense_seperate_gating_ln: bool = False
  dynamic_dense_share_qk_way: bool = False # default: share q/k/v with mlp way cross-layer composition
  dynamic_dense_fix_last_layer: bool = False # reduce static dense and dynamic dense to 1-way
  dynamic_dense_keep_residual: bool = False
  dynamic_dense_k_from_res: bool = False
  v_out_rank: Optional[int] = None
  v_out_dynamic: bool = False
  attn_out_orig: bool = False
  use_dense_norm: bool = False
  dense_norm_tpl: LayerTpl = template_field(normalizations.RmsNormNoScale)  # mqy
  dynamic_dense_act_cls: activations_lib.BaseActivation = None # mqy
  comp_dense_diff: bool = False  # mqy; compose difference of hidden state in every layer 
  dense_bias_init_method: Optional[str] = 'current_only' # emb_only, current_only, uniform
  dynamic_head_dense: bool = False
  dynamic_head_rank: int = 2
  dynamic_head_dense_type: str = 'qk'
  # dynamic_qk_proj: bool = False
  dynamic_head_seperate_param: bool = False
  dynamic_token_shift: int = 0
  laurel_lr: bool = False
  laurel_rw: bool = False
  laurel_normed_residual: bool = False
  gating_mlp_input: Optional[str] = None # ['sigmoid', 'linear', 'attn_out_depend+sigmoid'] (a, 1-a) or (a,b) 
  gating_mlp_input_act: activations_lib.BaseActivation = activations_lib.GELU
  use_mamba: bool = False
  mamba_tpl: LayerTpl = template_field(Mamba2) 



  # This function can be overridden by subclasses.
  def _setup_attention(self, atten_tpl: LayerTpl, name: str)-> None:
    atten_tpl = atten_tpl.clone()
    if name == 'self_attention':
      atten_tpl.name = 'multihead_self_atten'
    elif name == 'cross_attention':
      atten_tpl.name = 'multihead_cross_atten'
    else:
      atten_tpl.name = name
    atten_tpl.input_dim = self.input_dims
    atten_tpl.hidden_dim = self.num_heads * self.dim_per_head \
      if self.dim_per_head is not None else self.input_dims  # XD
    atten_tpl.num_heads = self.num_heads
    atten_tpl.dim_per_head = self.dim_per_head
    atten_tpl.atten_dropout_prob = self.atten_dropout_prob
    if self.ngrammer_tpl and name == 'self_attention':
      atten_tpl.ngrammer_tpl = self.ngrammer_tpl
    self.create_child(name, atten_tpl)

  def setup(self) -> None:

    # Initialize Layer Norm
    if self.norm_policy == 'primer_hybrid':
      params = self.ln_tpl.clone()
      params.dim = self.input_dims
      self.create_child('pre_layer_norm', params)
      self.create_child('post_layer_norm', params)
    elif self.norm_policy in ('pre', 'post', 'post_skip'):
      params = self.ln_tpl.clone()
      params.name = 'layer_norm'
      params.dim = self.input_dims
      self.create_child('layer_norm', params)
      if self.dynamic_dense_type is not None and self.dynamic_dense_sep_qkv_ln: # default init 3 layernorms 
        num_lns = sum([self.dynamic_dense_type.count(s) for s in 'qkv']) 
        assert self.dynamic_dense_type.endswith('m') # qkvm, kvm, qkm, qvm
        if not self.dynamic_dense_share_qk_way: 
          num_lns = min(3, num_lns+1) # Due to sharing one-way cross-layer composition with mlp, we need to create one more layernorm when num_qkv <3 
        self.create_children('layer_norms', [self.ln_tpl.clone().set(name=f'layer_norm_{cidx}', dim=self.input_dims) for cidx in range(num_lns)])
    else:
      raise ValueError('Unrecognized norm_policy: %s' % self.norm_policy)

    if self.use_mamba:
      assert self.norm_policy == 'pre'
      self.create_child('mamba', self.mamba_tpl.clone())
    else:
      # Initialize multi-headed self-attention
      self._setup_attention(self.tr_atten_tpl, 'self_attention')

      # Initialize residual dropout.
      params = self.dropout_tpl.clone()
      params.keep_prob = 1.0 - self.residual_dropout_prob
      self.create_child('residual_dropout', params)

      # Initialize multi-headed cross-attention and layer norm.
      if self.use_cross_attention:
        if self.norm_policy in ('pre', 'post', 'post_skip'):
          params = self.ln_tpl.clone()
          params.name = 'cross_layer_norm'
          params.dim = self.input_dims
          self.create_child('cross_layer_norm', params)
        elif self.norm_policy == 'primer_hybrid':
          params = self.ln_tpl.clone()
          params.dim = self.input_dims
          self.create_child('pre_cross_layer_norm', params)
          self.create_child('post_cross_layer_norm', params)
        else:
          raise ValueError(f'Unrecognized norm_policy: {self.norm_policy}')

        if self.cross_atten_tpl is not None:
          params = self.cross_atten_tpl
        else:
          params = self.tr_atten_tpl
        self._setup_attention(params, 'cross_attention')

      # Initialize residual droppath
      if self.residual_droppath_prob > 0:
        droppath_p = pax_fiddle.Config(
            stochastics.StochasticResidual,
            name='residual_droppath',
            survival_prob=1.0 - self.residual_droppath_prob,
        )
        self.create_child('residual_droppath', droppath_p)

      # Initialize feed-forward layer
      if self.tr_fflayer_tpl:
        if self.num_ffn == 1:
          params = self.tr_fflayer_tpl.clone()
          params.name = 'tr_fflayer'
          params.input_dims = self.input_dims
          params.hidden_dims = self.hidden_dims
          params.relu_dropout_prob = self.relu_dropout_prob
          params.residual_dropout_prob = self.residual_dropout_prob
          params.residual_droppath_prob = self.residual_droppath_prob
          params.norm_policy = self.norm_policy
          params.add_skip_connection = not self.gpt_j_residual  # XD
          params.seperate_gating_ln = self.dynamic_dense_seperate_gating_ln # mqy
          self.create_child('ff_layer', params)
        else:
          params_list = []
          for i in range(self.num_ffn):
            params = self.tr_fflayer_tpl.clone()
            params.name = f'tr_fflayer_{i}'
            params.input_dims = self.input_dims
            params.hidden_dims = self.hidden_dims
            params.relu_dropout_prob = self.relu_dropout_prob
            params.residual_dropout_prob = self.residual_dropout_prob
            params.residual_droppath_prob = self.residual_droppath_prob
            params.norm_policy = self.norm_policy
            params.add_skip_connection = False, 
            params_list.append(params) 
          self.create_children('ff_layers', params_list)

    
    # Intialize dense conn params
    if self.dense_conn: # default 
      i = self.layer_index
      params = self.dense_norm_tpl.clone()
      params.name = 'dense_norm'
      if hasattr(params, 'dim'): params.dim = self.input_dims
      self.create_child('dense_norm', params)
      # self.create_child('dense_norm', self.dense_norm_tpl.clone())
      if self.dynamic_dense_act_cls is not None:
        self.create_child('dense_activation', pax_fiddle.Config(self.dynamic_dense_act_cls).clone())
        # assert isinstance(self.dynamic_dense_act_cls, GELU)
      # assert self.dense_bias_init_method in ['current_only', 'emb_only', 'uniform']
      # b_init_method = self.dense_bias_init_method if not self.comp_dense_diff else 'uniform' # use uniform init with comp_dense_diff
        # if b_init_method == 'uniform':
        #   init_v = [1] * (i+2) 
        # elif b_init_method == 'current_only':
        #   init_v = [0] * (i+1) + [1] 
        # elif  b_init_method == 'emb_only':
        #   init_v = [1] + [0] * (i+1) 
        # dense_w = WeightHParams(
        #     shape=[len(init_v)],
        #     init=WeightInit.Constant(init_v), # L
        #     mesh_shape=self.mesh_shape,
        #     tensor_split_dims_mapping=[None],
        #     collections=[base_layer.WeightHParamsCollection.SKIP_LP_REGULARIZATION],  # XD
        # )
        # self.create_variable(f'dense_conn_{i}', dense_w)
      if self.dense_conn_on_attn: 
        assert self.dynamic_dense_query_wise and not self.dynamic_dense_key_wise
        factor = 2 if not self.attn_out_orig else 1
      elif self.dense_conn_on_layerdiff:
        factor = 2
      else: # default 
        factor = 1
      if self.dynamic_dense: # default: BTD, DL-> BTL
        if self.dynamic_dense_query_wise and self.dynamic_dense_key_wise: 
          l = i + 2 + (self.num_layers - i)
        elif self.dynamic_dense_query_wise: # default
          l = (i + 1) * factor + 1
        elif self.dynamic_dense_key_wise:
          l = self.num_layers - i
        if self.dynamic_dense_type is not None: # default
          C = 1 if self.dynamic_dense_fix_last_layer and i== self.num_layers-1 else len(self.dynamic_dense_type)
        else:
          C = 1 
        l = l * self.dynamic_dense_num_groups * C  # G=1
        std = 1/math.sqrt(self.input_dims)  
        dynamic_dense_inter_dim = int(l * self.dynamic_dense_hidden_expand)
        if self.dynamic_dense_hidden_round:  # default: round to 64 or 128
          # assert dynamic_dense_inter_dim < 128
          dynamic_dense_inter_dim = (dynamic_dense_inter_dim// 64 +1) * 64 
        dyn_dense_proj1 = WeightHParams(
            shape=[self.input_dims, dynamic_dense_inter_dim], # DL
            init=WeightInit.Gaussian(std),
            mesh_shape=self.mesh_shape, 
            tensor_split_dims_mapping=['data', None],
        )
        dyn_dense_proj2 = WeightHParams(
            shape=[dynamic_dense_inter_dim, l], # LL
            init=WeightInit.Constant(0),
            mesh_shape=self.mesh_shape, 
            tensor_split_dims_mapping=[None, None],
        )
        self.create_variable(f'dynamic_dense_conn1_{i}', dyn_dense_proj1)
        self.create_variable(f'dynamic_dense_conn2_{i}', dyn_dense_proj2)
        if self.dynamic_dense_gate_mlp: # ignore 
          assert self.dynamic_dense_hidden_expand == 1
          dyn_dense_proj_gate = WeightHParams(
              shape=[self.input_dims, l], # DL
              init=WeightInit.Constant(0),
              mesh_shape=self.mesh_shape, 
              tensor_split_dims_mapping=['data', None],
          )
          self.create_variable(f'dynamic_dense_gate_{i}', dyn_dense_proj_gate)
        elif self.dynamic_dense_glu_mlp: # ignore 
          dyn_dense_proj_gate = WeightHParams(
              shape=[self.input_dims, l* self.dynamic_dense_hidden_expand], # DL
              init=WeightInit.Gaussian(std),
              mesh_shape=self.mesh_shape, 
              tensor_split_dims_mapping=['data', None],
          )
          self.create_variable(f'dynamic_dense_gate_{i}', dyn_dense_proj_gate)

    if self.hyper_conn:
      self.create_child('hyper_conn_norm', self.hyper_conn_norm_tpl.clone().set(dim=self.input_dims))
      # static w: alpha_sw, beta_sw
      m_w = np.array([1 if _i == self.layer_index % self.hyper_conn_n else 0 for _i in range(self.hyper_conn_n)]) 
      r_w = np.eye(self.hyper_conn_n)
      sw_arr = np.concatenate([m_w[:,None], r_w], axis=-1)
      alpha_sw = WeightHParams(
          shape=[self.hyper_conn_n, 1+self.hyper_conn_n], # N(1+N)
          init=WeightInit.Constant(sw_arr.tolist()), 
          mesh_shape=self.mesh_shape,
          tensor_split_dims_mapping=[None, None],
          collections=[base_layer.WeightHParamsCollection.SKIP_LP_REGULARIZATION], 
      )
      self.create_variable('hyper_conn_alpha_sw', alpha_sw)
      beta_sw = WeightHParams(
          shape=[self.hyper_conn_n], # N
          init=WeightInit.Constant(1), 
          mesh_shape=self.mesh_shape,
          tensor_split_dims_mapping=[None],
          collections=[base_layer.WeightHParamsCollection.SKIP_LP_REGULARIZATION], 
      )
      self.create_variable('hyper_conn_beta_sw', beta_sw)
      # dynamic alpha w: alpha_dw, alpha_scale
      # std = 1/ math.sqrt(self.input_dims)
      alpha_dw = WeightHParams(
          shape=[self.input_dims, 1+self.hyper_conn_n], # D(1+N)
          # init=WeightInit.Gaussian(std),
          init=WeightInit.Constant(0),
          mesh_shape=self.mesh_shape, 
          tensor_split_dims_mapping=['data', None],
      )
      alpha_scale = WeightHParams(
          shape=[1],
          init=WeightInit.Constant(0.01), # L or CL
          mesh_shape=self.mesh_shape,
          tensor_split_dims_mapping=[None],
      )
      self.create_variable('hyper_conn_alpha_dw', alpha_dw)
      self.create_variable('hyper_conn_alpha_scale', alpha_scale)
      # dynamic beta w: beta_dw, beta_scale
      # std = 1/ math.sqrt(self.input_dims)
      beta_dw = WeightHParams(
          shape=[self.input_dims], # D
          # init=WeightInit.Gaussian(std),
          init=WeightInit.Constant(0),
          mesh_shape=self.mesh_shape, 
          tensor_split_dims_mapping=[None],
      )
      beta_scale = WeightHParams(
          shape=[1],
          init=WeightInit.Constant(0.01), # L or CL
          mesh_shape=self.mesh_shape,
          tensor_split_dims_mapping=[None],
      )
      self.create_variable('hyper_conn_beta_dw', beta_dw)
      self.create_variable('hyper_conn_beta_scale', beta_scale)


    if self.laurel_rw:
        init_v = [1, 1]
        rw = WeightHParams(
            shape=[len(init_v)],
            init=WeightInit.Constant(init_v), # L
            mesh_shape=self.mesh_shape,
            tensor_split_dims_mapping=[None],
            collections=[base_layer.WeightHParamsCollection.SKIP_LP_REGULARIZATION],  # XD
        )
        self.create_variable('laurel_rw_weight', rw)
    if self.laurel_lr:
      self.create_child('laurel_norm', self.dense_norm_tpl.clone())

    if self.dynamic_dense_ov or self.laurel_lr:
      ov_rank = self.dynamic_dense_ov_rank 
      std = 1/math.sqrt(self.input_dims)
      assert self.dynamic_dense_ov_init in ['gauss+gauss', 'gauss+const0']
      if self.dynamic_dense_ov_init == 'gauss+gauss':
        init_method = WeightInit.Gaussian(std) 
      elif self.dynamic_dense_ov_init == 'gauss+const0':
        init_method = WeightInit.Constant(0)
      dyn_ov1 = WeightHParams(
        shape=[self.input_dims, ov_rank], # DR
        init=WeightInit.Gaussian(std),
        mesh_shape=self.mesh_shape, 
        tensor_split_dims_mapping=['data', None],
      )
      dyn_ov2 = WeightHParams(
        shape=[ov_rank, self.input_dims], # RD
        init=init_method,
        mesh_shape=self.mesh_shape, 
        tensor_split_dims_mapping=[None, 'data'],
      )
      self.create_variable('dynamic_dense_ov1', dyn_ov1)
      self.create_variable('dynamic_dense_ov2', dyn_ov2)
      if self.dynamic_dense_ov_gate:
        dyn_ov_gate = WeightHParams(
        shape=[self.input_dims, ov_rank], # DR
        init=WeightInit.Gaussian(std),
        mesh_shape=self.mesh_shape, 
        tensor_split_dims_mapping=['data', None],
        )
        self.create_variable('dynamic_dense_ov3', dyn_ov_gate)

    if self.gating_mlp_input:
      self.create_child('gating_mlp_norm', self.dense_norm_tpl.clone())
      if self.gating_mlp_input_act is not None:
        self.create_child('gmi_activation', pax_fiddle.Config(self.gating_mlp_input_act).clone()) # gelu
      if 'learnable_bias' in self.gating_mlp_input:
        res_attn_bias = WeightHParams(
          shape=[2], # D2
          init=WeightInit.Constant(1),
          mesh_shape=self.mesh_shape, 
          tensor_split_dims_mapping=[None],
        )
        self.create_variable('res_attn_bias', res_attn_bias)
      if 'linear' in self.gating_mlp_input or 'sigmoid' in self.gating_mlp_input or 'b_only' in self.gating_mlp_input:
        res_attn_w = WeightHParams(
          shape=[self.input_dims, 2], # D2
          init=WeightInit.Constant(0),
          mesh_shape=self.mesh_shape, 
          tensor_split_dims_mapping=['data', None],
        )
      elif 'mlp' in self.gating_mlp_input:
        std = 1/math.sqrt(self.input_dims)
        res_attn_w = WeightHParams(
          shape=[self.input_dims, 2], # D2
          init=WeightInit.Gaussian(std),
          mesh_shape=self.mesh_shape, 
          tensor_split_dims_mapping=['data', None],
        )
        res_attn_w2 = WeightHParams(
          shape=[2, 2], # 22
          init=WeightInit.Constant(0),
          mesh_shape=self.mesh_shape, 
          tensor_split_dims_mapping=[None, None],
        )
        self.create_variable('res_attn_w2', res_attn_w2)
      self.create_variable('res_attn_w', res_attn_w)



    if self.dynamic_head_dense or self.v_out_rank: # TODO: disentangle config between dynamic_dense and dynamic_head_dense  
      i = self.layer_index
      params = self.dense_norm_tpl.clone()
      params.name = 'head_dense_norm'
      if hasattr(params, 'dim'): params.dim = self.input_dims
      self.create_child('head_dense_norm', params)
      if self.dynamic_dense_act_cls is not None:
        self.create_child('head_dense_activation', pax_fiddle.Config(self.dynamic_dense_act_cls).clone())

    if self.dynamic_head_dense:
      std = 1/math.sqrt(self.input_dims) 
      l = i+1
      n, R = self.num_heads, self.dynamic_head_rank
      r = self.v_out_rank if self.v_out_rank else n
      C = len(self.dynamic_head_dense_type)
      _l = l * C if self.dynamic_head_seperate_param else l
      K = _l * R * r + C * R * n 
      # K = C * R * n 
      std2 = 1 / math.sqrt(K) 
      dyn_head_dense_proj1 = WeightHParams(
          shape=[self.input_dims, K], # D(L+1)NR; BTD, DK->BTK , (L+C)BSRN
          init=WeightInit.Gaussian(std),
          mesh_shape=self.mesh_shape, 
          tensor_split_dims_mapping=['data', None], 
      )
      dyn_head_dense_proj2a = WeightHParams(
          shape=[K, _l, R, r], # K(L+1)CRr; BTK, K(LC)Rr->(LC)BTRr
          # init=WeightInit.Constant(0),
          init=WeightInit.Gaussian(std2),
          mesh_shape=self.mesh_shape, 
          tensor_split_dims_mapping=[None, None, None, None], # TODO: check sharding
      )
      dyn_head_dense_proj2b = WeightHParams(
          shape=[K, C, R, n], # K(L+1)RN; BTK, KLRN->LBTRN
          init=WeightInit.Constant(0),
          mesh_shape=self.mesh_shape, 
          tensor_split_dims_mapping=[None, None, None, None], # TODO: check sharding
      )
      self.create_variable(f'dynamic_head_dense_conn1_{i}', dyn_head_dense_proj1)
      self.create_variable(f'dynamic_head_dense_conn2a_{i}', dyn_head_dense_proj2a)
      self.create_variable(f'dynamic_head_dense_conn2b_{i}', dyn_head_dense_proj2b)
    if self.v_out_rank: 
      if self.v_out_dynamic: 
        K = self.num_heads * self.v_out_rank
        std1 = 2 / math.sqrt(self.input_dims + K)
        std2 = 1 / math.sqrt(K)
        v_out_dyn1 = WeightHParams(
        shape=[self.input_dims, K], # DK
        init=WeightInit.Gaussian(std1),
        mesh_shape=self.mesh_shape, 
        tensor_split_dims_mapping=[None, None], # TODO: check sharding
        )
        v_out_dyn2 = WeightHParams(
        shape=[K, self.num_heads, self.v_out_rank], # KNr
        init=WeightInit.Gaussian(std2),
        mesh_shape=self.mesh_shape, 
        tensor_split_dims_mapping=[None, None, None], # TODO: check sharding
        )
        self.create_variable('v_out_dyn1', v_out_dyn1)
        self.create_variable('v_out_dyn2', v_out_dyn2)
      else:
        std = 2 / math.sqrt(self.num_heads + self.v_out_rank)
        v_out_proj = WeightHParams(
        shape=[self.num_heads, self.v_out_rank], # Nr
        init=WeightInit.Gaussian(std),
        mesh_shape=self.mesh_shape, 
        tensor_split_dims_mapping=[None, None], # TODO: check sharding
        )
        self.create_variable('v_out_proj', v_out_proj)

    if self.dynamic_token_shift >= 2:
      self.create_child('token_shift_norm', self.dense_norm_tpl.clone())
      if self.dynamic_dense_act_cls is not None:
        self.create_child('token_shift_activation', pax_fiddle.Config(self.dynamic_dense_act_cls).clone())
      std = 1/math.sqrt(self.input_dims) 
      tf_proj1 = WeightHParams(
          shape=[self.input_dims, self.dynamic_token_shift], # Dt
          init=WeightInit.Gaussian(std),
          mesh_shape=self.mesh_shape, 
          tensor_split_dims_mapping=['data', None],
      )
      std2 = 1/math.sqrt(self.dynamic_token_shift)
      tf_proj2 = WeightHParams(
          shape=[self.dynamic_token_shift, self.dynamic_token_shift], # tt
          init=WeightInit.Gaussian(std2),
          # init=WeightInit.Constant(0),
          mesh_shape=self.mesh_shape, 
          tensor_split_dims_mapping=[None, None],
      )
      self.create_variable('dynamic_tf1', tf_proj1)
      self.create_variable('dynamic_tf2', tf_proj2)


  def init_states(self, target_batch_size: int, target_max_length: int) -> None:
    """Initialize the cache for the Transformer layer.

    Args:
      target_batch_size: Batch size for the target.
      target_max_length: The length to decode the target.

    Returns:
      None.
    """
    raise NotImplementedError(type(self))

  def decoding_sequence_length(self) -> int:
    """Get the decoding sequence length."""
    return self.self_attention.decoding_state_sequence_length()

  def __call__(
      self,
      inputs: JTensor,
      paddings: JTensor,
      attention_mask: JTensor,
      cross_inputs: Optional[JTensor] = None,
      cross_attention_mask: Optional[JTensor] = None,
      segment_pos: Optional[JTensor] = None,
      segment_ids: Optional[JTensor] = None,
      v_in:Optional[JTensor] = None, # mqy: cross head values, 2BTNd (2:q/k) 
      hhids:Optional[JTensor] = None, # mqy: hyper hidden states, NBTD (N:4) 
      residual:Optional[JTensor] = None, # original residual for qvm ablation where k uses residual stream directly
      ) -> Tuple[JTensor, JTensor]:
    """Transformer decoder layer.

    Args:
      inputs: Input sequence JTensor of shape [B, T, H].
      paddings: Input paddings JTensor of shape [B, T] (only used in FFN layer).
      attention_mask: Self attention mask ready to add to the logits. It can be
        of shape [1|B, 1, 1|T, T] which is broadcast compatible with the self
        attention matrix of shape [B, N, T, T]. This is assumed to have combined
        paddings, causal masking as well as segment maskings.
      cross_inputs: Output of the encoder, to be used for cross attention, of
        shape [B, S, H].
      cross_attention_mask: Cross attention mask ready to add to the logits. It
        can be of shape [1|B, 1, 1|T, S] which is broadcast compatible with the
        cross attention matrix of shape [B, N, T, S]. This is assumed to have
        combined paddings as well as segment maskings.
      segment_pos: A JTensor of shape [B, T]. The position of each token in a
        segment.
      segment_ids: A JTensor of shape [B, T] specifying which segment each token
        belongs to.

    Returns:
      The fflayer output with shape [B, T, D].
      atten_probs: A NestedMap with keys `self_atten` <float>[B, N, T, T].
    """

    # XD remove
    # inputs_stats = stats.compute_stats(inputs, jnp.expand_dims(paddings, -1))
    # self.add_summary('xformer_input_mean', inputs_stats.mean_v, verbosity=3)
    # self.add_summary('xformer_input_std', inputs_stats.std_v, verbosity=3)
    # self.add_summary('xformer_input_abs_max', inputs_stats.max_v, verbosity=3)

    # self.add_summary('attention_input_rms', _rms(inputs), verbosity=4)

    if self.hyper_conn and not self.hyper_conn_merge_wcdc: # ignore:  width connections
      alpha_sw, alpha_dw, alpha_scale = getattr(self.theta, 'hyper_conn_alpha_sw'), getattr(self.theta, 'hyper_conn_alpha_dw'), getattr(self.theta, 'hyper_conn_alpha_scale')
      beta_sw, beta_dw, beta_scale = getattr(self.theta, 'hyper_conn_beta_sw'), getattr(self.theta, 'hyper_conn_beta_dw'), getattr(self.theta, 'hyper_conn_beta_scale')
      hhids_normed = self.hyper_conn_norm(hhids)
      B_w = jnp.einsum('NBTD, D->BTN', hhids_normed, beta_dw) # beta 
      Amr_w = jnp.einsum('NBTD, DM->BTNM', hhids_normed, alpha_dw) # alpha  BTN(1+N)
      if self.hyper_conn_tanh: 
        B_w = jnp.tanh(B_w)
        Amr_w = jnp.tanh(Amr_w)
      Amr_w = Amr_w * alpha_scale + alpha_sw[None,None]
      B_w = B_w * beta_scale + beta_sw[None,None]
      Am_w, Ar_w = Amr_w[...,0], Amr_w[...,1:] # BTN, BTNN
      inputs = jnp.einsum('NBTD,BTN->BTD', hhids, Am_w) # mixed inputs
      hhids = jnp.einsum('NBTD,BTNM->MBTD',hhids,Ar_w) # mixed hyper hidden states
      
    if self.norm_policy == 'primer_hybrid':
      inputs_normalized = self.pre_layer_norm(inputs)
    elif self.norm_policy == 'pre':
      if self.dynamic_dense_stack or self.layer_index ==0: # lidx==0 tensor
        inputs_normalized = self.layer_norm(inputs) # BTD; CBTD 
      else: # lidx>0 tuple
        assert isinstance(inputs, tuple)
        if self.dynamic_dense_type is not None and _contains_one(self.dynamic_dense_type, ['qkv','kvm', 'qkm', 'qvm']): # default: q,k,v,[q/k/v][q/k/v], qkv
          # num_qkv = sum([self.dynamic_dense_type.count(s) for s in 'qkv']) 
          if self.dynamic_dense_sep_qkv_ln: # default
            assert self.dynamic_dense_type.endswith('m')
            inputs_normalized = []
            for _type in 'qkv':
              if _type in self.dynamic_dense_type: # default
                _idx = self.dynamic_dense_type.index(_type)
              elif self.dynamic_dense_share_qk_way and _type in ['q', 'k']:
                _idx = self.dynamic_dense_type.index('qk'.replace(_type, '')) # q->k or k->q
              else: 
                _idx = -1 # share mlp-way cross-layer composition or apply ln to residual 
              if self.dynamic_dense_k_from_res:
                assert residual is not None
              _input = residual if self.dynamic_dense_k_from_res and _idx == -1 else inputs[_idx]  # default q/k/v hidden state else
              inputs_normalized.append(self.layer_norms[_idx](_input))
          else: 
            inputs_normalized = [self.layer_norm(_inputs) for _inputs in inputs]
        else: # dynamic_dense_type = 'm' 
          assert len(inputs) == 2 # residual_stream_orig, one_way_dynamic_dense 
          inputs_normalized = self.layer_norm(inputs[0])
    else:
      inputs_normalized = inputs

    if self.dynamic_dense_type is not None and self.layer_index > 0 : # default
      dyn_inputs = inputs
      inputs = inputs[-1] if self.dynamic_dense_type.endswith('m') else inputs[0]
    if self.use_mamba:
      output = self.mamba(inputs_normalized) + inputs
      atten_probs, v_out = None, None
    else:
      ### beginning of attn_mlp
      # Compute self-attention, key/value vectors are the input itself
      if self.dynamic_dense_type is not None and _contains_one(self.dynamic_dense_type, ['qkv','kvm','qkm','qvm']) and self.layer_index >0: # default lidx>0
        query_vec, key_vec, value_vec = inputs_normalized[0], inputs_normalized[1], inputs_normalized[2]
      else: # lidx==0
        query_vec, key_vec, value_vec = inputs_normalized, inputs_normalized, inputs_normalized
      atten_output, self_atten_probs, v_out = self.self_attention(
          query_vec,
          key_vec,
          value_vec,
          atten_mask=attention_mask,
          query_segment_pos=segment_pos,
          key_segment_pos=segment_pos,
          v_in=v_in)
      atten_output_orig = atten_output
      atten_probs = NestedMap(self_atten=self_atten_probs)

      self.add_summary('attention_output_rms', _rms(atten_output), verbosity=4)

      if self.norm_policy == 'primer_hybrid':
        atten_output = self.post_layer_norm(atten_output)
      elif self.norm_policy == 'post':
        atten_output = self.layer_norm(atten_output)

      self.add_summary('attention_output_norm_rms', _rms(atten_output),
                      verbosity=4)

      # Residual dropout and connection
      atten_output = self.residual_dropout(atten_output)

      # Apply skip connection
      if self.residual_droppath_prob > 0.0:
        atten_output = self.residual_droppath(inputs, atten_output)
      else: 
        if self.dynamic_dense_type is not None and self.dynamic_dense_type.endswith('m') and self.layer_index > 0: # lidx>0
          res = dyn_inputs[-1]
        else: # lidx==0
          res = inputs
        atten_output += res

      if self.norm_policy == 'post_skip':
        atten_output = self.layer_norm(atten_output)

      # self.add_summary('attention_output_rel_cos', _rel_cos(inputs, atten_output),
      #                 verbosity=4)

      # Apply cross attention if applicable
      if self.use_cross_attention and (
          not self.allow_skip_cross_attention or cross_inputs is not None
      ):
        assert cross_inputs is not None
        assert cross_attention_mask is not None
        if self.norm_policy == 'pre':
          atten_output_normalized = self.cross_layer_norm(atten_output)
        elif self.norm_policy == 'primer_hybrid':
          atten_output_normalized = self.pre_cross_layer_norm(atten_output)
        elif self.norm_policy in ('post', 'post_skip'):
          atten_output_normalized = atten_output

        cross_atten_output, cross_atten_probs = self.cross_attention(
            atten_output_normalized,
            cross_inputs,
            cross_inputs,
            atten_mask=cross_attention_mask)
        atten_probs.cross_atten = cross_atten_probs

        if self.norm_policy == 'post':
          cross_atten_output = self.cross_layer_norm(cross_atten_output)
        elif self.norm_policy == 'primer_hybrid':
          cross_atten_output = self.post_cross_layer_norm(cross_atten_output)

        # Residual dropout and connection
        cross_atten_output = self.residual_dropout(cross_atten_output)

        if self.residual_droppath_prob > 0.0:
          atten_output = self.residual_droppath(atten_output, cross_atten_output)
        else:
          atten_output += cross_atten_output

        if self.norm_policy == 'post_skip':
          atten_output = self.cross_layer_norm(atten_output)

      # gating mlp inputs
      if self.gating_mlp_input:
        if 'attn_out_depend' in self.gating_mlp_input:
          _input = atten_output
        else: 
          _input = inputs
        res_attn_w = jnp.einsum('B T D, D R -> B T R', self.gating_mlp_norm(_input), self.theta.res_attn_w) # 
        if 'learnable_bias' in self.gating_mlp_input:
          bs = self.theta.res_attn_bias 
        else:
          bs = [1, 1]
        if 'linear' in self.gating_mlp_input:
          a, b = res_attn_w[:,:,:1]+bs[0], res_attn_w[:,:,1:2]+bs[1] # 1 as bias, 
        elif 'sigmoid' in self.gating_mlp_input:
          a = jax.nn.sigmoid(res_attn_w[:,:,:1]) * 2
          b = 2 - a
        elif 'mlp' in self.gating_mlp_input:
          res_attn_w = jnp.einsum('B T R, R Q -> B T Q', self.gmi_activation(res_attn_w), self.theta.res_attn_w2) # 
          a, b = res_attn_w[:,:,:1]+bs[0], res_attn_w[:,:,1:2]+bs[1]
        elif 'b_only' in self.gating_mlp_input:
          a, b = 1, res_attn_w[:,:,1:2]+bs[1]
        mlp_input = inputs * a + (atten_output - inputs) * b 
      else:
        mlp_input = atten_output

      # Apply FFN layer
      if self.num_ffn == 1: # default
        gate_inputs = None 
        if self.layer_index >0 and self.dynamic_dense_type is not None and 'g' in self.dynamic_dense_type:
          gate_inputs = atten_output_orig + dyn_inputs[self.dynamic_dense_type.index('g')]
        output = self.ff_layer(mlp_input, paddings=paddings, gate_inputs=gate_inputs) \
          if not self.gpt_j_residual else atten_output + self.ff_layer(inputs, paddings=paddings)  # XD
        if self.dynamic_dense_type is not None and 'l' in self.dynamic_dense_type and self.layer_index !=0: # ignore
          output = (output - mlp_input) + atten_output_orig + dyn_inputs[self.dynamic_dense_type.index('l')] # -2 for 'qkvlm'
      else:
        assert self.dynamic_dense_type is not None and self.dynamic_dense_type.count('m') == self.num_ffn
        if self.layer_index !=0:
          mlp_inputs = [atten_output_orig + _mlp_input for _mlp_input in dyn_inputs[-self.num_ffn:]]
        else:
          mlp_inputs = [mlp_input] * self.num_ffn 
        output = atten_output + sum([ff_layer(mlp_input, paddings=paddings) for ff_layer, mlp_input in zip(self.ff_layers, mlp_inputs)]) / 4
      
      
      if not self.gpt_j_residual and self.gating_mlp_input and 'skip_gating_residual' in self.gating_mlp_input:
        output = (output - mlp_input) + atten_output
      self.add_summary('layer_output_rms', _rms(output-inputs), verbosity=4)
      ### end of attn _mlp 

    # output = inputs + (atten_output - inputs) + (output - atten_output)
    if self.laurel_lr:
      if self.laurel_rw:
        rw = jax.nn.softmax(self.theta.laurel_rw_weight, axis=0) * 2 # [1,1] -> [0.5, 0.5] * 2 -> [1,1]
      else:
        rw = [1, 1]
      inputs_normed = self.laurel_norm(inputs)
      inner = jnp.einsum('B T D, D R -> B T R', inputs_normed, self.theta.dynamic_dense_ov1)
      output_lr = jnp.einsum('B T R, R D -> B T D', inner, self.theta.dynamic_dense_ov2)
      if self.laurel_normed_residual:
        output_lr = output_lr + inputs_normed
      output = inputs * rw[0] + (output - inputs) * rw[1] + output_lr


    # generate dynamic dense conn weight
    dyn_dense_w = None
    if self.dynamic_dense: # default 
      dyn_dense_proj = (getattr(self.theta, f'dynamic_dense_conn1_{self.layer_index}'),  getattr(self.theta, f'dynamic_dense_conn2_{self.layer_index}'))
      dense_proj1, dense_proj2 = dyn_dense_proj
      x_out_normed = self.dense_norm(output) if self.use_dense_norm else output
      if self.dynamic_dense_act_cls is None:
        dense_proj = dense_proj1 @ dense_proj2 if self.dynamic_dense_multilayer else dense_proj1 # DL,LL->DL # 1024x25 , 25x25 -> 1024x25
        dyn_dense_w = jnp.einsum('B T D, D L -> B T L', x_out_normed, dense_proj)
      else: # default
        dense_w_inner = self.dense_activation(jnp.einsum('B T D, D K -> B T K', x_out_normed, dense_proj1)) # GELU
        if self.dynamic_dense_gate_mlp:
          gate_w = getattr(self.theta, f'dynamic_dense_gate_{self.layer_index}') # DL
          gate = jnp.einsum('B T D, D K -> B T K', x_out_normed, gate_w) # BTL
          dyn_dense_w = gate * dense_w_inner
        elif self.dynamic_dense_glu_mlp:
          gate_w = getattr(self.theta, f'dynamic_dense_gate_{self.layer_index}') # DL
          gate = jnp.einsum('B T D, D K -> B T K', x_out_normed, gate_w) # BTL
          dyn_dense_w = jnp.einsum('B T K, K L -> B T L', gate * dense_w_inner, dense_proj2)
        else: # default 
          dyn_dense_w = jnp.einsum('B T K, K L -> B T L', dense_w_inner, dense_proj2)

    if self.dynamic_dense_ov: # ignore 
      assert self.dynamic_dense
      if not self.dynamic_dense_ov_outer_loop:
        inner = jnp.einsum('B T D, D R -> B T R', output, self.theta.dynamic_dense_ov1)
        output = output + jnp.einsum('B T R, R D -> B T D', inner, self.theta.dynamic_dense_ov2)
        if self.laurel_normed_residual: output = output + x_out_normed
      if self.dynamic_dense and self.dynamic_dense_ov_outer_loop: 
        dyn_dense_w = (dyn_dense_w, self.theta.dynamic_dense_ov1, self.theta.dynamic_dense_ov2, (self.theta.dynamic_dense_ov3 if self.dynamic_dense_ov_gate else None))      


    dyn_head_dense_w = None   
    if self.dynamic_head_dense or self.v_out_rank: # ignore 
      x_out_normed = self.head_dense_norm(output) if self.use_dense_norm else output

    if self.dynamic_head_dense: # ignore 
      dyn_head_dense_proj = (getattr(self.theta, f'dynamic_head_dense_conn1_{self.layer_index}'),  getattr(self.theta, f'dynamic_head_dense_conn2a_{self.layer_index}'), getattr(self.theta, f'dynamic_head_dense_conn2b_{self.layer_index}'))
      head_dense_proj1, head_dense_proj2a, head_dense_proj2b = dyn_head_dense_proj
      dense_w_inner = self.head_dense_activation(jnp.einsum('BTD,DK->BTK', x_out_normed, head_dense_proj1))   # GELU activation
      if self.v_out_rank is None:
        head_dense_proj2 = jnp.concatenate([head_dense_proj2a, head_dense_proj2b], axis=1) # KLRN, KCRN->K(L+C)RN
        dyn_head_dense_w = jnp.einsum('BTK,KLRN->LBTRN', dense_w_inner, head_dense_proj2) # (L*C+C)BTRN
      else:
        dyn_head_dense_wa = jnp.einsum('BTK,KLRQ->LBTRQ', dense_w_inner, head_dense_proj2a) # BTK, KLRr->LBTRr
        dyn_head_dense_wb = jnp.einsum('BTK,KCRN->CBTRN', dense_w_inner, head_dense_proj2b) # 
        dyn_head_dense_w = (dyn_head_dense_wa, dyn_head_dense_wb)

    if self.v_out_rank: # ignore 
      if self.v_out_dynamic:
        _v_out_proj = self.head_dense_activation(jnp.einsum('BTD,DK->BTK', x_out_normed, self.theta.v_out_dyn1))   # GELU activation
        v_out_proj = jnp.einsum('BTK,KNR->BTNR', _v_out_proj, self.theta.v_out_dyn2)
        v_out = jnp.einsum('BTND, BTNR -> BTRD', v_out, v_out_proj)
      else:
        v_out = jnp.einsum('BTND, NR -> BTRD', v_out, self.theta.v_out_proj)

    if self.dynamic_token_shift >= 2:
      x_out_normed = self.token_shift_norm(output)
      tf_inner = self.token_shift_activation(jnp.einsum('BTD,DK->BTK', x_out_normed, self.theta.dynamic_tf1))   # GELU activation
      tf_w = jnp.einsum('BTK,KF->BTF', tf_inner, self.theta.dynamic_tf2) # BTF
      assert self.dynamic_token_shift == 2
      _output = jnp.zeros(output.shape, dtype=output.dtype)
      ouput = output + _output.at[:, 1:, :].set(output[:,:-1] * tf_w[:,1:,:1]) + output * tf_w[:,:,1:]
    attn_out = atten_output_orig if self.attn_out_orig else atten_output
    
    if self.hyper_conn:
      if not self.hyper_conn_merge_wcdc: # depth connections 
        hhids = hhids +  jnp.einsum('BTD,BTN->NBTD',output - inputs, B_w)  # fuse current output into hyper hidden states
      else:
        alpha_sw, alpha_dw, alpha_scale = getattr(self.theta, 'hyper_conn_alpha_sw'), getattr(self.theta, 'hyper_conn_alpha_dw'), getattr(self.theta, 'hyper_conn_alpha_scale')
        beta_sw, beta_dw, beta_scale = getattr(self.theta, 'hyper_conn_beta_sw'), getattr(self.theta, 'hyper_conn_beta_dw'), getattr(self.theta, 'hyper_conn_beta_scale')
        hhids_normed = self.hyper_conn_norm(hhids)
        B_w = jnp.einsum('NBTD, D->BTN', hhids_normed, beta_dw) # beta 
        Amr_w = jnp.einsum('NBTD, DM->BTNM', hhids_normed, alpha_dw) # alpha  BTN(1+N)
        if self.hyper_conn_tanh: 
          B_w = jnp.tanh(B_w)
          Amr_w = jnp.tanh(Amr_w)
        Amr_w = Amr_w * alpha_scale + alpha_sw[None,None]
        B_w = B_w * beta_scale + beta_sw[None,None]
        Am_w, Ar_w = Amr_w[...,0], Amr_w[...,1:] # BTN, BTNN
        hhids = jnp.einsum('NBTD,BTNM->MBTD',hhids,Ar_w) +  jnp.einsum('BTD,BTN->NBTD',output - inputs, B_w)  # fuse current output into hyper hidden states
        output = jnp.einsum('NBTD,BTN->BTD', hhids, Am_w)

    return output, attn_out, atten_probs, dyn_dense_w, dyn_head_dense_w, v_out, hhids  # pytype: disable=bad-return-type  # jax-ndarray

  def extend_step(self,
                  inputs: JTensor,
                  *,
                  time_step: JTensor,
                  segment_pos: Optional[JTensor] = None,
                  attention_mask: JTensor,
                  cross_attention_mask: Optional[JTensor] = None) -> JTensor:
    # pyformat:disabled
    """Transformer decoder layer, autoregressive cached decoding.

    For cross attention, the key/value cache may have a smaller batch size b
    than inputs batch size B. In this case, we require B % b == 0, and this
    corresponds to multi-sample decoding for each input in b, and cross-
    attention states will be repeated by (B // b) times. Each consecutive
    (B // b) chunk in B correspond to multiple samples for the same cross
    # inputs.

    When `inputs` has shape [B, D], it will do extend_step on one token per
    batch in regular autoregressive decoding.

    When `inputs` has shape [B, L, D], it will do extend_step on L tokens per
    batch. This is used to do suffix scoring after autoregressive decoding.

    Args:
      inputs:         [B, D] or [B, L, D], target sequence at index time_step.
      time_step:      a 0-based scalar, the current decode step.
      segment_pos:    [B] or [B, L], the current position in the same segment.
        If unspecified, time_step will be used.

      attention_mask: [B, 1, L, S] if extends multiple steps (i.e. `inputs` is
        of shape [B, L, D]) or [B, 1, T] if extends one step (i.e. `inputs` is
        of shape [B, D]), optional attention mask for this time step. This
        combines causal mask with any segment mask if applicable.
      cross_attention_mask: [b|B, 1, 1 S], optional, cross_segment_mask for
        this time step. This combines padding mask with any segment mask if
        applicable.

    Returns:
      output: [B, D] or [B, L, D].
    """
    # Layer normalize input
    if self.norm_policy == 'primer_hybrid':
      inputs_normalized = self.pre_layer_norm(inputs)
    elif self.norm_policy == 'pre':
      inputs_normalized = self.layer_norm(inputs)

    # Self-attention layer.
    atten_output = self.self_attention.extend_step(
        inputs_normalized,
        atten_mask=attention_mask,
        time_step=time_step,
        segment_pos=segment_pos)
    if self.norm_policy == 'primer_hybrid':
      atten_output = self.post_layer_norm(atten_output)
    elif self.norm_policy == 'post':
      atten_output = self.layer_norm(atten_output)

    # Residual dropout and connection
    atten_output = self.residual_dropout(atten_output)
    atten_output += inputs

    # Apply cross attention if applicable
    if self.use_cross_attention and (
        not self.allow_skip_cross_attention or cross_attention_mask is not None
    ):
      assert cross_attention_mask is not None
      if self.norm_policy == 'pre':
        atten_output_normalized = self.cross_layer_norm(atten_output)
      elif self.norm_policy == 'primer_hybrid':
        atten_output_normalized = self.pre_cross_layer_norm(atten_output)
      elif self.norm_policy == 'post':
        atten_output_normalized = atten_output

      cross_atten_output = self.cross_attention.extend_step(
          atten_output_normalized,
          atten_mask=jnp.squeeze(cross_attention_mask, 2),
          time_step=time_step,
          segment_pos=segment_pos,
          is_cross_attention=True)

      if self.norm_policy == 'post':
        cross_atten_output = self.cross_layer_norm(cross_atten_output)
      elif self.norm_policy == 'primer_hybrid':
        cross_atten_output = self.post_cross_layer_norm(cross_atten_output)

      # Residual dropout and connection
      cross_atten_output = self.residual_dropout(cross_atten_output)
      atten_output += cross_atten_output

    # Apply FFN layer
    output = self.ff_layer.extend_step(atten_output, time_step=time_step)
    return output

  def transform_decode_state(
      self, transform_fn: base_layer.DecodeStateTransformFn) -> None:
    """Transforms all decode state variables based on transform_fn."""
    self.self_attention.transform_decode_state(transform_fn)

  def lazy_broadcast_prefix(self, num_suffix_samples: int,
                            suffix_length: int) -> None:
    """Performs lazy prefix broadcast on the decoding states.

    Current decoding states will be moved to PREFIX_DECODE_CACHE. New decoding
    state will be created for the suffixes with multiple samples sharing
    previous prefixes.

    Args:
      num_suffix_samples: Number of samples that will share the same previous
        decoding state.
      suffix_length: The length of the new suffix samples.
    """
    self.self_attention.lazy_broadcast_prefix(num_suffix_samples, suffix_length)

  def right_align_decode_state_with_prefix(
      self, max_prefix_size: int,
      right_align_fn: base_layer.DecodeStateTransformFn) -> None:
    """Right aligns decode state with prefix decode states."""
    self.self_attention.right_align_decode_state_with_prefix(
        max_prefix_size, right_align_fn)

def wsum(w: jnp.ndarray, # CBTL1  # XD
         hids: list[jnp.ndarray], # list of BTD
         seq_chunk_size: int = None
         ) -> jnp.ndarray:  # CBTD
  C, B, T, L, _ = w.shape; D = hids[0].shape[-1]
  out = jnp.zeros((C, B, T, D), dtype=hids[0].dtype)
  seq_chunk_size = seq_chunk_size or T
  assert T % seq_chunk_size == 0
  for chunk_i in range(T // seq_chunk_size):
    sli = slice(chunk_i * seq_chunk_size, (chunk_i + 1) * seq_chunk_size)
    for l in range(L):
      if seq_chunk_size == D:
        out += w[:, :, :, l, :] * hids[l]
      else:
        # out[:, :, sli, :] += w[:, :, sli, l, :] * hids[l][:, sli, :]  # CBt1*BtD->CBtD)
        out = out.at[:, :, sli, :].set(out[:, :, sli, :] + w[:, :, sli, l, :] * hids[l][:, sli, :])  # CBt1*BtD->CBtD)
  return out if out.shape[0] > 1 else jnp.squeeze(out, 0)  # squeeze C dim if C == 1

class StackedTransformer(base_layer.BaseLayer):
  """A stack of Transformer layers.

  Attributes:
    use_cross_attention: If True, introduces cross encoder-decoder attention
      layer.
    mask_self_attention: Use masked self-attention.
    num_layers: Number of layers in this stack.
    model_dims: Model dimension in Transformer layers.
    hidden_dims: The hidden layer dimension of FFN in Transformer layers.
    num_heads: Number of attention heads.
    dim_per_head: Dimension of each attention head. If None then dim_per_head ==
      model_dims // num_heads.
    dropout_prob: Apply dropout at this prob at various places.
    residual_droppath_prob: Probability at which we drop the entire residual
      path.
    input_dropout_prob: Dropout probability applied to the input before any
      processing happens.
    gating_func: Gating function type--can be one of the following options:
      'top2', based on the GShard paper: https://arxiv.org/abs/2006.16668,
      'expert_choice', based on https://arxiv.org/abs/2202.09368, 'dense_top2':
      experimental gating function for decodiing. Similar to 'top2' gating, but
      no capacity constrainst for each expert.
    unadjusted_expert_capacity_factor: Unadjusted expert capacity_factor. This
      is the ratio between global batch size and total capacity across all
      experts and all routing groups.
    transformer_layer_params_tpl: A template of Transformer.params, can be a
      list of params of length equal to the num_layers or a factor of
      num_layers. For a factor, the params are tiled as [a, a, ..., b, b,...,].
    packed_input: If True, each training example may pack multiple sequences.
    fold_padding_with_segment_mask: If True then segment mask is supposed to
      include the padding mask as well, i.e. treating PADs as one sequence and
      non-PADs as another.
    moe_layer_tpl: Template configuration for the moe feedforward layer.
    num_experts: Total number of experts.
    num_groups: Number of groups for dispathcing.
    min_group_size: If not None, num_groups will be adjusted so that there will
      be at least min_group_size tokens in each group.
    moe_layers: List of MoE layer indices, e.g. [0, 2, 4].
    ngrammer_tpls: Sequence of params for the Ngrammer layer. This param is
      shared between the Ngrammer layer as well as the VQNgrammer layer. The
      length of the sequence must match the number of attention layers. If an
      entry in the sequence is None, then there is no NGrammer layer present in
      that corresponding layer.
    remat: Boolean, whether to remat each layer to save memory.
    checkpoint_policy: How to checkpoint residuals for BProp: save nothing, dot
      only or dot with no batch dimensions.
  """
  use_cross_attention: bool = False
  mask_self_attention: bool = False
  num_layers: int = 0
  model_dims: int = 0
  hidden_dims: Union[int, Tuple[int]] = 0
  num_heads: int = 0
  dim_per_head: Optional[int] = None
  dropout_prob: float = 0.0
  atten_dropout_prob: Optional[float] = None
  residual_dropout_prob: Optional[float] = None
  relu_dropout_prob: Optional[float] = None
  residual_droppath_prob: float = 0.0
  input_dropout_prob: float = 0.0
  gating_func: str = 'top2'
  unadjusted_expert_capacity_factor: float = 2.0
  transformer_layer_params_tpl: Union[LayerTpl, Sequence[LayerTpl]] = (
      template_field(Transformer)
  )
  packed_input: bool = False
  fold_padding_with_segment_mask: bool = False
  moe_layer_tpl: Optional[LayerTpl] = template_field(TransformerFeedForwardMoe)
  num_experts: int = 0
  num_groups: int = 1
  min_group_size: Optional[int] = None
  moe_layers: Optional[Sequence[int]] = ()
  ngrammer_tpls: Optional[Sequence[LayerTpl]] = template_field(None)
  remat: bool = False
  checkpoint_policy: AutodiffCheckpointType = (
      AutodiffCheckpointType.SAVE_DOT_EXCEPT_LOGITS_FFN1
  )
  share_interval: Optional[int] = None # 2 for interleave sharing, 1 for sharing all layers
  share_interval_idxs: Optional[list] = None
  share_mode : str = 'interleave' # interleave or immediate 
  share_attn_only: bool = False
  share_qknorm: bool = True
  share_qkov: bool = True
  share_dynamic_proj: bool = True
  share_except_layers: Optional[list] = None
  use_slope_rate: bool = False
  slope_rate_lidxs: Optional[list] = None 
  lrpe_layers: Optional[list] = None
  pre_compute_atten_mask: bool = True
  hyper_conn: bool = False 
  hyper_conn_merge_wcdc: bool = False # whether to merge width connection and depth connection
  hyper_conn_n: int = 4
  hyper_conn_tanh: bool = False
  hyper_conn_norm_tpl: LayerTpl = template_field(normalizations.RmsNorm)  # mqy
  dense_conn: bool = False
  dense_conn_on_attn: bool = False
  dense_conn_on_layerdiff: bool = False
  dense_conn_learnable: bool = True
  dense_finetune_scale: bool = False
  dense_vector_scale: bool= False
  dense_vector_scale_keywise: bool = False
  num_ffn: int = 1 # mqy; group of feedforward layer
  dynamic_dense: bool = False
  dynamic_dense_ft_norm: bool = False
  dynamic_dense_sep_qkv_ln: bool = False
  dynamic_dense_stack: bool = True
  dynamic_dense_type: Optional[str] = None # ['qkv', 'qkvm']
  dynamic_dense_normalized: bool = False # softmax dynamic dense weight along the Layer dim 
  dynamic_dense_num_groups : int = 1
  dynamic_dense_ov: bool = False
  dynamic_dense_ov_gate: bool = False
  dynamic_dense_ov_init: str = 'gauss+gauss' # ['gauss+gauss', 'gauss+const0']
  dynamic_dense_ov_outer_loop: bool = False
  dynamic_dense_ov_rank: int = 64
  dynamic_dense_ov_after_merge: bool = False 
  dynamic_dense_hidden_expand: Union[int, list] = 1
  dynamic_dense_hidden_round: bool = False
  dynamic_dense_key_wise: bool = False 
  dynamic_dense_query_wise: bool = True
  dynamic_dense_multilayer: bool = True
  dynamic_dense_gate_mlp: bool = False
  dynamic_dense_glu_mlp: bool = False  
  dynamic_dense_norm_on_weight: bool = False 
  dynamic_dense_disentangle: bool = False 
  dynamic_dense_seperate_gating_ln: bool = False
  dynamic_dense_share_qk_way: bool = False # default: share q/k/v with mlp way cross-layer composition
  dynamic_dense_fix_last_layer: bool = False # reduce multi-way static and dynamic dense to 1-way
  dynamic_dense_by_group_heads: bool = False
  dynamic_dense_keep_residual: bool = False
  dynamic_dense_k_from_res: bool = False
  dynamic_dense_param_residual: bool = False
  ddense_w_gen_pattern: str = ''
  ddense_w_gen_chunk_size: int = None
  v_out_rank: Optional[int] = None
  v_out_dynamic: bool = False
  attn_out_orig: bool = False
  use_dense_norm: bool = False
  dense_norm_tpl: LayerTpl = template_field(normalizations.RmsNormNoScale)  # mqy
  dynamic_dense_act_cls: activations_lib.BaseActivation = None # mqy
  comp_dense_diff: bool = False  # mqy; compose difference of hidden state in every layer 
  dense_bias_init_method: Optional[str] = 'current_only' # emb_only, current_only, uniform
  dynamic_head_dense: Union[bool, tuple] = False
  dynamic_head_rank: Union[int, tuple] = 2 # 
  head_dw1_norm_on_activation: bool = True
  head_dw1_norm_tpl: LayerTpl = template_field(normalizations.RmsNormNoScale)  # mqy
  dynamic_head_dense_type: str = 'qk'
  dynamic_head_seperate_param: bool = False
  # dynamic_qk_proj: bool = False
  laurel_lr: bool = False
  laurel_rw: bool = False
  laurel_normed_residual: bool = False
  use_recurrent_layer_mixing: bool = False
  recurrent_layer_mixing_tpl: LayerTpl = template_field(rnn_cell.RecurrentLayerMix) # rnn, lstm 
  mamba_lidxs: Optional[list] = None # list(range(48))
  use_mamba: bool = False



  def _clone_layer_params(self, layer_tpl: LayerTpl) -> LayerTpl:
    """Useful to let sublasses switch the class (e.g. Streaming version)."""
    return layer_tpl.clone()

  def setup(self) -> None:

    assert self.num_layers > 0
    assert self.model_dims > 0
    # from collections.abc import Iterable
    # assert self.hidden_dims > 0 or isinstance(self.hidden_dims, Iterable)
    # assert self.num_heads > 0
    assert 0.0 <= self.dropout_prob < 1.0
    assert 0.0 <= self.input_dropout_prob < 1.0

    def _layer_params(i):
      """Construct i-th layer params."""
      if isinstance(self.transformer_layer_params_tpl, Sequence):
        factor = self.num_layers // len(self.transformer_layer_params_tpl)
        ii = i // factor
        p_i = self._clone_layer_params(self.transformer_layer_params_tpl[ii])
      else:
        p_i = self._clone_layer_params(self.transformer_layer_params_tpl)
      p_i.name = f'layer_{i}'

      # set mamba layer params
      if self.use_mamba:
        mamba_lidxs = self.mamba_lidxs if self.mamba_lidxs is not None else list(range(self.num_layers))
        p_i.use_mamba = i in mamba_lidxs  
        p_i.mamba_tpl.layer_idx = i 
        p_i.mamba_tpl.d_model = self.model_dims

      share_except_layers = self.share_except_layers if self.share_except_layers is not None else []
      if self.share_interval is not None and i not in share_except_layers:
        assert self.share_mode in ['immediate', 'interleave']
        if self.share_mode == 'interleave':
          ii = i % self.share_interval
        elif self.share_mode == 'immediate':
          ii = i // self.share_interval
        
        if self.share_interval_idxs is None: 
          if self.share_mode == 'interleave':
            ii_max = self.share_interval 
          elif self.share_mode == 'immediate':
            ii_max = self.num_layers // self.share_interval
          share_interval_idxs = list(range(ii_max))
        else:
          share_interval_idxs = self.share_interval_idxs

        if ii in share_interval_idxs:  # specify shared idxs in the block
          if self.share_attn_only:
            if self.share_qknorm and self.share_qkov and self.share_dynamic_proj:
              p_i.tr_atten_tpl.shared_weight_layer_id = f'shared_attn_{ii}'
            else: #TOOD(mqy):add other tpls for sharing 
              shared_tpls = []
              #shared_tpls = ['cross_head_pre_proj_tpl', 'cross_head_post_proj_tpl', 'dynamic_w_pre_proj_tpl', 'dynamic_w_post_proj_tpl','proj_tpl', 'qk_norm_tpl']
              if self.share_qknorm: shared_tpls += ['qk_norm_tpl']
              if self.share_dynamic_proj: shared_tpls += ['cross_head_pre_proj_tpl', 'cross_head_post_proj_tpl', 'dynamic_w_pre_proj_tpl', 'dynamic_w_post_proj_tpl']
              if self.share_qkov: shared_tpls += ['proj_tpl']
              for name in shared_tpls:
                shared_name = 'shared_' + name.replace('_tpl', '') + f'_{ii}'
                setattr(getattr(p_i.tr_atten_tpl,name), 'shared_weight_layer_id', shared_name)
                logging.info(f'sid-{name}: {shared_name}')
          else:
            p_i.shared_weight_layer_id = f'shared_layer_{ii}'

      lrpe_layers = [] if self.lrpe_layers is None else self.lrpe_layers
      if i in lrpe_layers: p_i.tr_atten_tpl.use_lrpe = True  
      if self.use_slope_rate: # mqy for TransNormer
        def get_slopes_power_of_2(n):
          start = 2**(-(2**-(jnp.log2(n) - 3)))
          ratio = start
          return [start * ratio**i for i in range(n)]
        slope_rate = jnp.array(get_slopes_power_of_2(self.num_heads)) 
        if self.slope_rate_lidxs is None:
          slope_rate_lidx = i 
        else:
          assert len(self.slope_rate_lidxs) == self.num_layers
          slope_rate_lidx = self.slope_rate_lidxs[i] 
        p_i.tr_atten_tpl.slope_rate = slope_rate * (1 - slope_rate_lidx/(self.num_layers-1) + 1e-5)

      p_i.use_cross_attention = self.use_cross_attention
      p_i.num_heads = self.num_heads
      p_i.dim_per_head = self.dim_per_head
      p_i.input_dims = self.model_dims
      p_i.packed_input = self.packed_input
      p_i.atten_dropout_prob = self.atten_dropout_prob or self.dropout_prob
      p_i.residual_dropout_prob = (
          self.residual_dropout_prob or self.dropout_prob
      )
      p_i.relu_dropout_prob = self.relu_dropout_prob or self.dropout_prob
      p_i.hidden_dims = self.hidden_dims

      if self.residual_droppath_prob > 0.0:
        p_i.residual_droppath_prob = (
            self.residual_droppath_prob * i / max(1, self.num_layers)
        )
      # hyper connections params
      p_i.hyper_conn = self.hyper_conn 
      p_i.hyper_conn_n = self.hyper_conn_n
      p_i.hyper_conn_tanh = self.hyper_conn_tanh
      p_i.hyper_conn_norm_tpl = self.hyper_conn_norm_tpl
      p_i.hyper_conn_merge_wcdc = self.hyper_conn_merge_wcdc
      
      # dense params
      if isinstance(self.dynamic_dense_hidden_expand, tuple) or isinstance(self.dynamic_dense_hidden_expand, list):
        assert len(self.dynamic_dense_hidden_expand) == self.num_layers
        dynamic_dense_hidden_expand = self.dynamic_dense_hidden_expand[i]
      else:
        dynamic_dense_hidden_expand = self.dynamic_dense_hidden_expand
      p_i.layer_index = i
      p_i.num_ffn = self.num_ffn
      p_i.num_layers = self.num_layers
      p_i.dense_conn = self.dense_conn
      p_i.dense_conn_on_attn = self.dense_conn_on_attn
      p_i.dense_conn_on_layerdiff = self.dense_conn_on_layerdiff
      p_i.dynamic_dense = self.dynamic_dense
      p_i.dynamic_dense_sep_qkv_ln = self.dynamic_dense_sep_qkv_ln
      p_i.dynamic_dense_stack = self.dynamic_dense_stack
      p_i.dynamic_dense_type = self.dynamic_dense_type
      p_i.dynamic_dense_num_groups = self.dynamic_dense_num_groups
      p_i.dynamic_dense_ov = self.dynamic_dense_ov
      p_i.dynamic_dense_ov_gate = self.dynamic_dense_ov_gate
      p_i.dynamic_dense_ov_init = self.dynamic_dense_ov_init
      p_i.dynamic_dense_ov_outer_loop = self.dynamic_dense_ov_outer_loop
      p_i.dynamic_dense_ov_rank = self.dynamic_dense_ov_rank
      p_i.dynamic_dense_hidden_expand = dynamic_dense_hidden_expand
      p_i.dynamic_dense_hidden_round = self.dynamic_dense_hidden_round
      p_i.dynamic_dense_key_wise = self.dynamic_dense_key_wise
      p_i.dynamic_dense_query_wise = self.dynamic_dense_query_wise
      p_i.dynamic_dense_multilayer = self.dynamic_dense_multilayer
      p_i.dynamic_dense_gate_mlp = self.dynamic_dense_gate_mlp
      p_i.dynamic_dense_glu_mlp = self.dynamic_dense_glu_mlp 
      p_i.dynamic_dense_seperate_gating_ln = self.dynamic_dense_seperate_gating_ln
      p_i.dense_bias_init_method = self.dense_bias_init_method
      p_i.comp_dense_diff = self.comp_dense_diff
      p_i.dynamic_dense_share_qk_way = self.dynamic_dense_share_qk_way
      p_i.dynamic_dense_fix_last_layer = self.dynamic_dense_fix_last_layer
      p_i.dynamic_dense_keep_residual = self.dynamic_dense_keep_residual
      p_i.dynamic_dense_k_from_res = self.dynamic_dense_k_from_res


      p_i.laurel_lr = self.laurel_lr
      p_i.laurel_rw = self.laurel_rw
      p_i.laurel_normed_residual = self.laurel_normed_residual

      # shared hyper-params by dense and head_dense
      p_i.use_dense_norm = self.use_dense_norm
      p_i.dynamic_dense_act_cls = self.dynamic_dense_act_cls

      # head dense params
      p_i.dynamic_head_dense = self.dynamic_head_dense[i] if isinstance(self.dynamic_head_dense, tuple) else self.dynamic_head_dense
      p_i.dynamic_head_rank = self.dynamic_head_rank[i] if isinstance(self.dynamic_head_rank, tuple) else self.dynamic_head_rank
      p_i.dynamic_head_dense_type = self.dynamic_head_dense_type
      p_i.tr_atten_tpl.dynamic_head_dense_type = self.dynamic_head_dense_type
      p_i.dynamic_head_seperate_param = self.dynamic_head_seperate_param
      p_i.v_out_rank = self.v_out_rank
      p_i.v_out_dynamic = self.v_out_dynamic
      p_i.attn_out_orig = self.attn_out_orig
      # p_i.dynamic_qk_proj = self.dynamic_qk_proj

      if self.moe_layers and i in self.moe_layers:
        assert self.num_experts > 0
        assert self.moe_layer_tpl is not None
        moe_p = self.moe_layer_tpl.clone()
        moe_p.num_experts = self.num_experts
        moe_p.num_groups = self.num_groups
        moe_p.min_group_size = self.min_group_size
        moe_p.gating_func = self.gating_func
        if moe_p.hidden_dims:
          # MoE hidden_dims could be different from FFN hidden_dims
          p_i.hidden_dims = moe_p.hidden_dims
        p_i.tr_fflayer_tpl = moe_p

      atten_tpl = p_i.tr_atten_tpl

      atten_tpl.dynamic_dense_by_group_heads = self.dynamic_dense_by_group_heads
      # assert atten_tpl.project_logits
      # assert atten_tpl.project_probs
      # if i == 1:
      #   logging.warning('In StackedTransformer.setup: disable cross_head_proj for layer 1')  # XD debug
      #   atten_tpl.project_probs = atten_tpl.project_logits = False
      if hasattr(atten_tpl, 'cross_head_pre_proj_tpl'):
        cross_head_proj_tpls = [atten_tpl.cross_head_pre_proj_tpl, atten_tpl.cross_head_post_proj_tpl]
      if hasattr(atten_tpl, 'dynamic_w_pre_proj_tpl'):
        cross_head_proj_tpls += [atten_tpl.dynamic_w_pre_proj_tpl, atten_tpl.dynamic_w_post_proj_tpl]
      for p in cross_head_proj_tpls:
        for name in ['use_static_w', 'dynamic_w_init', 'dynamic_d_init', 'src_dependent','tgt_dependent']:
          if hasattr(p, name):
            val = getattr(p, name)
            # if i > 0: assert isinstance(val, list), f'{p}.{name} = {val}'
            if isinstance(val, list):
              assert len(val) == self.num_layers, f'{len(val)} != {self.num_layers}'
              setattr(p, name, val[i])
      # Several attributes of atten_tpl (inc. window_size) have already beens set, but
      # num_heads and dim_per_head haven't, which are to be set in Transformer._setup_attention
      for tpl, tpl_name, names in [(p_i, 'p_i', ['hidden_dims', 'num_heads', 'dim_per_head']),  # tuple
                        (atten_tpl, 'atten_tpl', ['window_size', 'project_probs', 'project_logits', 'internal_enable_query_scale'] + ['o_norm', 'o_gate_activation_cls', 'use_rotary_position_emb', 'linear_attn', 'qk_activation_cls'] )]:  # list
        for name in names:
          if hasattr(tpl, name):
            val = getattr(tpl, name)
            if isinstance(val, (list, tuple)):  # TODO: When do type(hidden_dims/p_i.num_heads/dim_per_head)
                                                # turn from list to tuple? During repeat?
              # logging.info(f'In StackedTransformer.setup._layer_params: {tpl_name}.{name} = {val} {type(val)}')
              assert len(val) == self.num_layers, f'{len(val)} != {self.num_layers}'
              setattr(tpl, name, val[i])

      if self.ngrammer_tpls is not None:
        if self.ngrammer_tpls[i] is not None:
          p_i.ngrammer_tpl = self.ngrammer_tpls[i]
      return p_i

    if isinstance(self.transformer_layer_params_tpl, (list, tuple)):
      if self.num_layers % len(self.transformer_layer_params_tpl):
        raise ValueError('num_layers should be divisible by '
                         'transformer_layer_params_tpl')

    if isinstance(self.dynamic_head_rank, tuple):
      assert len(self.dynamic_head_rank) == self.num_layers
    if isinstance(self.dynamic_head_dense, tuple):
      assert len(self.dynamic_head_dense) == self.num_layers

    if self.dynamic_dense_type is not None:
      assert self.dynamic_dense_type in ['qkv', 'qkvm', 'm', 'qkvmm', 'qkvlm', 'qkvgm', 'kvm', 'qkm', 'qvm']

    layer_params = [_layer_params(i) for i in range(self.num_layers)]
    self.create_children('x_layers', layer_params)

    self.create_child('head_dw1_norm', self.head_dw1_norm_tpl.clone())

    if self.dense_conn:
      # self.create_child('dense_norm', self.dense_norm_tpl.clone())
        # assert isinstance(self.dynamic_dense_act_cls, GELU)
      assert self.dense_bias_init_method in ['current_only', 'emb_only', 'uniform', 'zeros3+current_only', 'current_only_softmax']
      if self.dense_conn_learnable: # default 
        b_init_method = self.dense_bias_init_method if not self.comp_dense_diff else 'uniform' # use uniform init with comp_dense_diff
        if self.dense_conn_on_attn or self.dense_conn_on_layerdiff:
          assert 'current_only' in b_init_method
          factor = 2 if not self.attn_out_orig else 1
        else: # default 
          factor = 1
        for i in range(self.num_layers):
          if self.dynamic_dense_type is not None: # default
            C = 1 if i==self.num_layers-1 and self.dynamic_dense_fix_last_layer else len(self.dynamic_dense_type)
          else:
            C = None 
          if b_init_method == 'uniform':
            init_v = [1] * (i+2) 
          elif b_init_method == 'current_only':  # default
            init_v = [0] * ((i+1) * factor) + [1] 
          elif b_init_method == 'emb_only':
            init_v = [1] + [0] * (i+1) 
          elif b_init_method == 'current_only_softmax':
            init_v = [0] * (i+1) + [5]
          elif b_init_method == 'zeros3+current_only':
            init_v = [0] * ((i+1) * factor) + [1]
            init_vs = [ [0] * len(init_v) ] * 3 + [init_v]
          dense_w = WeightHParams(
              shape=[len(init_v)] if C is None else [C, len(init_v)],  # default else [C, len(init_v)]
              init=WeightInit.Constant(init_vs if b_init_method == 'zeros3+current_only' else init_v), # L or CL # default init_v
              mesh_shape=self.mesh_shape,
              tensor_split_dims_mapping=[None] if C is None else [None, None], # default else
              collections=[base_layer.WeightHParamsCollection.SKIP_LP_REGULARIZATION],  # XD
          )
          self.create_variable(f'dense_conn_{i}', dense_w)

      if self.dense_vector_scale: # ignore
        assert self.dynamic_dense_type is not None
        num_vecs = self.num_layers +1 if self.dense_vector_scale_keywise else self.num_layers
        for i in range(num_vecs):
          C = 1 if i==self.num_layers-1 and self.dynamic_dense_fix_last_layer and not self.dense_vector_scale_keywise else len(self.dynamic_dense_type)
          vector_scale = WeightHParams(
              shape=[C, self.model_dims], # D 
              init=WeightInit.Constant(1),
              mesh_shape=self.mesh_shape,
              tensor_split_dims_mapping=[None, None],
              collections=[base_layer.WeightHParamsCollection.SKIP_LP_REGULARIZATION],  # XD
          )
          self.create_variable(f'dense_vector_scale_{i}', vector_scale)

      if self.dense_finetune_scale: # ignore
        for i in range(self.num_layers+1):
          ft_scale = WeightHParams(
              shape=[self.model_dims], # D 
              init=WeightInit.Constant(1),
              mesh_shape=self.mesh_shape,
              tensor_split_dims_mapping=[None],
              collections=[base_layer.WeightHParamsCollection.SKIP_LP_REGULARIZATION],  # XD
          )
          self.create_variable(f'dense_ft_scale_{i}', ft_scale)

      if self.dynamic_dense_key_wise: # ignore: BTD, DL-> BTL
        if self.use_dense_norm: 
          self.create_child('emb_dense_norm', self.dense_norm_tpl.clone())
        if self.dynamic_dense_act_cls is not None:
          self.create_child('emb_dense_activation', pax_fiddle.Config(self.dynamic_dense_act_cls).clone())
        std = 1/math.sqrt(self.model_dims) 
        l = self.num_layers * self.dynamic_dense_num_groups 
        dyn_dense_proj1 = WeightHParams(
            shape=[self.model_dims, l * self.dynamic_dense_hidden_expand], # DL
            init=WeightInit.Gaussian(std),
            mesh_shape=self.mesh_shape, 
            tensor_split_dims_mapping=['data', None],
        )
        dyn_dense_proj2 = WeightHParams(
            shape=[l * self.dynamic_dense_hidden_expand, l], # LL
            init=WeightInit.Constant(0),
            mesh_shape=self.mesh_shape, 
            tensor_split_dims_mapping=[None, None],
        )
        self.create_variable('emb_dynamic_dense_conn1', dyn_dense_proj1)
        self.create_variable('emb_dynamic_dense_conn2', dyn_dense_proj2)

    if self.use_recurrent_layer_mixing:
      self.create_child('layer_mix', self.recurrent_layer_mixing_tpl.clone())

    if self.input_dropout_prob > 0.0:
      self.create_child(
          'input_dropout',
          pax_fiddle.Config(
              stochastics.Dropout, keep_prob=1.0 - self.input_dropout_prob
          ),
      )

  def init_states(self, *args: Any, **kwargs: Any) -> None:
    """Initialize the cache for the StackedTransformer layer.

    Args:
      *args: Other arguments.
      **kwargs: Other keyword arguments.

    Returns:
      None.
    """
    raise NotImplementedError(type(self))

  def __call__(self,
               inputs: JTensor,
               paddings: JTensor,
               segment_mask: Optional[JTensor] = None,
               cross_inputs: Optional[JTensor] = None,
               cross_paddings: Optional[JTensor] = None,
               cross_segment_mask: Optional[JTensor] = None,
               segment_pos: Optional[JTensor] = None) -> JTensor:
    """Stacked Transformer layer.

    Args:
      inputs: Input sequence of shape [B, T, H].
      paddings: Input paddings of shape [B, T].
      segment_mask: Segment mask for packed input of shape [B, 1, T, T] ready to
        add to logits.
      cross_inputs: Output of the encoder, to be used for cross attention, of
        shape [B, S, H].
      cross_paddings: Paddings for cross atention of shape [B, S].
      cross_segment_mask: Segment mask for encoder-decoder in packed input case
        of shape [B, 1, T, S].
      segment_pos: Segment pos for packed input of shape [B, T].

    Returns:
      Output vector with shape [B, T, D].
    """
    if self.packed_input:
      assert segment_mask is not None

    if self.use_cross_attention:
      assert cross_inputs is not None
      assert cross_paddings is not None
      if self.packed_input:
        assert cross_segment_mask is not None

    attention_mask, cross_attention_mask = compute_attention_masks_for_fprop(
        inputs,
        paddings,
        self.mask_self_attention,
        segment_mask,
        cross_inputs,
        cross_paddings,
        cross_segment_mask,
        fold_padding_with_segment_mask=self.fold_padding_with_segment_mask,
    )

    x_out = inputs
    if self.input_dropout_prob > 0.0:
      x_out = self.input_dropout(x_out)

    def _fprop(
        transformer,
        x_in,
        paddings,
        attention_mask,
        cross_inputs,
        cross_attention_mask,
        segment_pos,
        v_in,
        hhids,
        residual,
    ):
      x_out, atten_output, _, dyn_dense_w, dyn_head_dense_w, v_out, hhids = transformer(
          x_in,
          paddings,
          attention_mask,
          cross_inputs,
          cross_attention_mask,
          segment_pos=segment_pos,
          v_in=v_in, # ignore
          hhids=hhids, # ignore
          residual=residual, # ignore
      )
      # dyn_dense_w = None
      # if dyn_dense_proj is not None:
      #   dense_proj1, dense_proj2 = dyn_dense_proj
      #   # x_out_normed = self.dense_norm(x_out) if self.use_dense_norm else x_out
      #   if self.dynamic_dense_act_cls is None:
      #     dense_proj = dense_proj1 @ dense_proj2 # DL,LL->DL # 1024x25 , 25x25 -> 1024x25
      #     dyn_dense_w = jnp.einsum('B T D, D L -> B T L', x_out, dense_proj)
      #   else:
      #     # dense_w_inner = self.dense_activation(jnp.einsum('B T D, D K -> B T K', x_out, dense_proj1))   # GELU activation
      #     dense_w_inner = jax.nn.gelu(jnp.einsum('B T D, D K -> B T K', x_out, dense_proj1))   # GELU activation
      #     dense_w_inner = self.dense_norm(dense_w_inner)
      #     dyn_dense_w = jnp.einsum('B T K, K L -> B T L', dense_w_inner, dense_proj2)

      #   # dyn_dense_w = jnp.einsum('B T D, D L -> B T L', self.dense_norm(x_out), dyn_dense_proj)
      #   # dyn_dense_w = self.dense_norm(dyn_dense_w)
      return x_out, atten_output, dyn_dense_w, dyn_head_dense_w, v_out, hhids # useful: x_out, dyn_dense_w: CBTL

    fprop = _fprop
    if self.remat:
      fprop = nn.remat(
          _fprop, policy=checkpoint_policy.custom_policy(self.checkpoint_policy)
      )
    
    if self.dense_conn: # default
      hids = [x_out]
      if self.dynamic_dense_key_wise: # ignore
        dense_proj1, dense_proj2 = self.theta.emb_dynamic_dense_conn1, self.theta.emb_dynamic_dense_conn2
        x_out_normed = self.emb_dense_norm(x_out) if self.use_dense_norm else x_out
        if self.dynamic_dense_act_cls is None:
          dense_proj = dense_proj1 @ dense_proj2 # DL,LL->DL # 1024x25 , 25x25 -> 1024x25
          dyn_dense_w_keywise = jnp.einsum('B T D, D L -> B T L', x_out_normed, dense_proj)
        else:
          dense_w_inner = self.emb_dense_activation(jnp.einsum('B T D, D K -> B T K', x_out_normed, dense_proj1))  # GELU activation
          dyn_dense_w_keywise = jnp.einsum('B T K, K L -> B T L', dense_w_inner, dense_proj2)
        dyn_dense_w_keywise = rearrange(dyn_dense_w_keywise, 'B T (L G) -> B T L G', G=self.dynamic_dense_num_groups)
        dws_key_wise = [dyn_dense_w_keywise] # BTLG, L=num_layers

    if self.hyper_conn:
      hhids = jnp.stack([x_out] * self.hyper_conn_n) # nD hyper hidden states
    else: # default
      hhids = None

    if self.use_recurrent_layer_mixing:
      h_0 = jnp.zeros(x_out.shape, dtype=self.fprop_dtype)
      cell_0 = jnp.zeros(x_out.shape, dtype=self.fprop_dtype)
      layer_hid, x_out, cell = self.layer_mix(x_out, h_0, cell_0)
    v_in = None # ignore
    v_outs = [] # ignore
    x_in = x_out
    last_dyn_dense_w = None # ignore
    residual = x_out if self.dynamic_dense_keep_residual else None # ignore 
    block_factor = 2 if self.dense_conn_on_attn or self.dense_conn_on_layerdiff else 1 # default 1 
    for i in range(self.num_layers):  
      hid_idxs = list(range((i+1)*block_factor +1)) # L+1
      # if self.dynamic_dense:
      #   dyn_dense_proj = (getattr(self.theta, f'dynamic_dense_conn1_{i}'),  getattr(self.theta, f'dynamic_dense_conn2_{i}'))
      # else:
      #   dyn_dense_proj = None 
      x_out, atten_output, dyn_dense_w, dyn_head_dense_w, v_out, hhids = fprop(
          self.x_layers[i],
          x_in, # lidx==0 array: BTD, lidx>0 tuple: [BTD]*4
          paddings,
          attention_mask,
          cross_inputs,
          cross_attention_mask,
          segment_pos,
          v_in,
          hhids,
          residual,
      )
      residual = x_out if self.dynamic_dense_keep_residual else None # ignore
      x_out = checkpoint_name(x_out, 'transformer_layer_out')
      v_in = None
      if self.dynamic_dense_disentangle: # ignore
        x_out = (x_out - x_in) + hids[-1]
      if self.use_recurrent_layer_mixing: # ignore
        layer_hid, x_out, cell = self.layer_mix(x_out, layer_hid, cell)

      if self.dense_conn: # default: update x_in for next layer
        dense_w = getattr(self.theta, f'dense_conn_{i}') if self.dense_conn_learnable else jnp.array([0] * (i+1) + [1]) 
        if self.dense_conn_on_attn: # ignore
          hids.append(atten_output)
        elif self.dense_conn_on_layerdiff: # ignore
          if self.dynamic_dense_type is not None and self.dynamic_dense_type.endswith('m'):
            hids.append(x_out - x_in[-1])
          else:
            hids.append(x_out - x_in)
        if self.comp_dense_diff: 
          if self.dynamic_dense_type is not None and self.dynamic_dense_type.endswith('m'):
            hids.append(x_out - x_in[-1])
          else:
            hids.append(x_out - x_in)
        else: # default 
          hids.append(x_out)
        if self.dynamic_dense:
          # dyn_dense_w = jnp.einsum('B T D, D L -> B T L', x_out, dyn_dense_proj) 
          # dyn_dense_w = base_layer.maybe_shard(dyn_dense_w, ['data', None, None], self.mesh_axis_names)
          ov = None # ignore 
          if isinstance(dyn_dense_w, tuple): # ignore 
            dyn_dense_w, ov = dyn_dense_w[0], dyn_dense_w[1:] # ov = ov1, ov2
          
          if self.dynamic_dense_norm_on_weight: # ignore
            assert self.dynamic_dense_num_groups == 1 and self.use_dense_norm
            dyn_dense_w = self.x_layers[i].dense_norm(dyn_dense_w) # BTL 2-25 

          if self.dynamic_dense_ov: # ignore
            assert ov is not None
          if not self.attn_out_orig: # ignore
            assert len(hids) == dense_w.shape[-1] and dyn_dense_w is not None
          
          if self.dynamic_dense_type is not None: # default 
            C = 1 if self.dynamic_dense_fix_last_layer and i==self.num_layers-1  else len(self.dynamic_dense_type) 
          if self.dynamic_dense_query_wise and self.dynamic_dense_key_wise:
            assert self.dynamic_dense_num_groups == 1 and not self.dynamic_dense_normalized
            assert self.num_layers + 2 == dyn_dense_w.shape[-1] / self.dynamic_dense_num_groups 
            dyn_dense_w = rearrange(dyn_dense_w, 'B T (L G) -> B T L G', G=self.dynamic_dense_num_groups)
            qw, kw = dyn_dense_w[:,:,:i+2],dyn_dense_w[:,:,i+2:]
            dws_key_wise.append(kw)
          elif self.dynamic_dense_query_wise: # default
            if self.dynamic_dense_type is not None: # default G=1
              dyn_dense_w = rearrange(dyn_dense_w, 'B T (C L G) -> C B T L G', G=self.dynamic_dense_num_groups, C=C)
            else: 
              assert len(hids) == dyn_dense_w.shape[-1] / self.dynamic_dense_num_groups
              dyn_dense_w = rearrange(dyn_dense_w, 'B T (L G) -> B T L G', G=self.dynamic_dense_num_groups)
          elif self.dynamic_dense_key_wise:
            assert self.dynamic_dense_num_groups == 1 and not self.dynamic_dense_normalized
            assert self.num_layers - (len(hids) - 2) == dyn_dense_w.shape[-1] / self.dynamic_dense_num_groups
            dyn_dense_w = rearrange(dyn_dense_w, 'B T (L G) -> B T L G', G=self.dynamic_dense_num_groups)
            dws_key_wise.append(dyn_dense_w) # BTLG; L=num_layers - i
          def _ov_transform(_x, ov, normalize=False): # ignore 
            if ov is None: # identity transformation
              return _x if not normalize else self.x_layers[i].dense_norm(_x) 
            else: # low_rank transformation
              ov1, ov2, ov_gate = ov 
              _layer = self.x_layers[i]
              _x_normed = _layer.dense_norm(_x) if _layer.use_dense_norm else _x
              _inner = jnp.einsum('B T D, D R -> B T R', _x_normed, ov1)
              if ov_gate is not None: 
                _inner = _inner * jnp.einsum('B T D, D R -> B T R', _x_normed, ov_gate)
              _output_lr = jnp.einsum('B T R, R D -> B T D', _inner, ov2)
              if self.laurel_normed_residual: _output_lr = _output_lr + _x_normed
              return _x + _output_lr
          if self.dynamic_dense_ov_after_merge:
            ov_before, ov_after = None, ov
          else: #default
            ov_before, ov_after = ov, None
          if self.dynamic_dense_num_groups == 1: # default
            if self.dynamic_dense_normalized:
              dyn_dense_w = jax.nn.softmax(dyn_dense_w + jnp.expand_dims(dense_w, 1), axis=2) # BTLG + LG-> BTLG 
              x_out = sum([dyn_dense_w[:,:,j] * _ov_transform(hids[j], ov_before) for j in hid_idxs])
            else: #default
              if self.dynamic_dense_query_wise and self.dynamic_dense_key_wise:
                x_out = sum([(dense_w[j] + qw[:,:,j] + dws_key_wise[j][:,:,self.num_layers-1-i]) * _ov_transform(hids[j], ov_before) for j in hid_idxs])
              elif self.dynamic_dense_query_wise: # default
                if self.dynamic_dense_type is not None: # default
                  if last_dyn_dense_w is not None and i < self.num_layers-1: # ignore 
                    dyn_dense_w = jnp.concatenate([dyn_dense_w[...,:-1,:] + last_dyn_dense_w, dyn_dense_w[...,-1:,:]], axis=-2)
                  if self.dynamic_dense_param_residual:  # ignore 
                    last_dyn_dense_w = dyn_dense_w
                  dyn_dense_w = dyn_dense_w + dense_w[:,None,None,:,None]  # dynamic + static; CBTL1 + C11L1 -> CBTL1
                  # C = len(self.dynamic_dense_type)
                  if self.dynamic_dense_type == 'qkvm' and self.attn_out_orig: # CBTL
                    assert self.dynamic_dense_type == 'qkvm'
                    x_out_qkv = [sum([dyn_dense_w[cidx,:,:,j] * hids[(j-1)*2+1] for j in range(1,i+2)]) + hids[-1] for cidx in range(3)] # 3BTL, LBTD -> 3BTD; L = num_layers  
                    x_out_m = sum([dyn_dense_w[3,:,:,j] * hids[j*2] for j in range(0,i+2)]) # BTL, LBTD -> BTD; L = num_layers + 1 
                    x_out_qkv.append(x_out_m)
                    x_out = jnp.stack(x_out_qkv)
                  elif self.dynamic_dense_type == 'm' and not self.dynamic_dense_stack:
                    x_out = (hids[-1], sum([dyn_dense_w[0,:,:,j] * _ov_transform(hids[j], ov_before) for j in hid_idxs]))
                  else: # default DyanmicDense ['qkvm', 'qkvgm', 'qkm', 'qvm', 'kvm'] branch
                    if self.dense_finetune_scale:
                      x_out = tuple([sum([dyn_dense_w[cidx,:,:,j] * (_ov_transform(hids[j], ov_before, normalize=self.dynamic_dense_ft_norm) * getattr(self.theta, f'dense_ft_scale_{j}')[None,None,:]) for j in hid_idxs]) for cidx in range(C)])
                    elif self.dense_vector_scale:
                      if self.dense_vector_scale_keywise:
                        x_out = tuple([sum([dyn_dense_w[cidx,:,:,j] * (_ov_transform(hids[j], ov_before) * getattr(self.theta, f'dense_vector_scale_{j}')[None,None,cidx,:]) for j in hid_idxs]) for cidx in range(C)])
                      else:
                        x_out = tuple([sum([dyn_dense_w[cidx,:,:,j] * (_ov_transform(hids[j], ov_before) * getattr(self.theta, f'dense_vector_scale_{i}')[None,None,cidx,:]) for j in hid_idxs]) for cidx in range(C)])
                    else: # default 
                      # _ov_transform return inputs 
                      if self.ddense_w_gen_pattern == '': # default
                        x_out = tuple([sum([dyn_dense_w[cidx,:,:,j] * _ov_transform(hids[j], ov_before, normalize=self.dynamic_dense_ft_norm) for j in hid_idxs]) for cidx in range(C)])
                      elif self.ddense_w_gen_pattern == 'q,k,v,m':
                        x_out = tuple([wsum(dyn_dense_w[cidx: cidx + 1], hids, self.ddense_w_gen_chunk_size) for cidx in range(C)])
                      elif self.ddense_w_gen_pattern == 'qkvm':
                        x_out = tuple(wsum(dyn_dense_w, hids, self.ddense_w_gen_chunk_size))
                      elif self.ddense_w_gen_pattern == 'qk,vm':
                        x_out = tuple(wsum(dyn_dense_w[:2], hids, self.ddense_w_gen_chunk_size)) + \
                          tuple(wsum(dyn_dense_w[2:], hids, self.ddense_w_gen_chunk_size)) \
                          if C > 1 else tuple(wsum(dyn_dense_w, hids, self.ddense_w_gen_chunk_size))
              
                    if self.dynamic_dense_stack: #ignore 
                      x_out = jnp.stack(x_out)
                else: # dynamic dense branch
                  dyn_dense_w = dyn_dense_w + dense_w[None,None,:,None]  # BTLG, L
                  x_out = sum([dyn_dense_w[:,:,j] * _ov_transform(hids[j], ov_before) for j in hid_idxs]) # BTL,LBTD-> BTD [(1+BT1->BT1), BTD->BTD for l in L]
                  # x_out = sum([(dense_w[j] + dyn_dense_w[:,:,j]) * _ov_transform(hids[j], ov_before) for j in hid_idxs]) # BTL,LBTD-> BTD [(1+BT1->BT1), BTD->BTD for l in L]
              elif self.dynamic_dense_key_wise:
                # dws_key_wise: [BTL,BTL,BT(L-1),...BT1] L=num_layers, len=25
                x_out = sum([(dense_w[j] + dws_key_wise[j][:,:,self.num_layers-1-i]) * _ov_transform(hids[j], ov_before) for j in hid_idxs])
          else:
            group_dim = self.model_dims//self.dynamic_dense_num_groups
            x_out = sum([jnp.repeat(dense_w[j] + dyn_dense_w[:,:,j], group_dim, axis=-1) * _ov_transform(hids[j], ov_before) for j in hid_idxs]) # 1 + BTLG -> BTLG -> BTL(Gd) repeat ;BTLD,LBTD-> BTD
          if self.dynamic_dense_ov_after_merge: # ignore 
            x_out = _ov_transform(x_out, ov_after)
          # if not self.dynamic_dense_key_wise:
          #   self.add_summary(f'dynamic_dense_w_max_{i}', dyn_dense_w.max(), verbosity=3)
          #   self.add_summary(f'dynamic_dense_w_mean_{i}', dyn_dense_w.mean(), verbosity=3)
          #   self.add_summary(f'dynamic_dense_w_min_{i}', dyn_dense_w.min(), verbosity=3)
        else:
          x_out = sum([dense_w[j] * hids[j] for j in range(len(hids))])
      if self.dynamic_head_dense: # ignore 
        v_outs.append(v_out)
      if self.x_layers[i].dynamic_head_dense: # ignore: update v_in for next layer
        C = len(self.dynamic_head_dense_type) # 2: qk, 3:qkv, 4: qkvo
        # dw2, dw1 = jnp.split(dyn_head_dense_w, jnp.array([2], dtype=jnp.int32), axis=0) # (L+1)BSRN -> 2BSRN,LBSRN
        if isinstance(dyn_head_dense_w, tuple):
          dw1, dw2 = dyn_head_dense_w # LBTRr, CBTRN
        else:
          dw1, dw2 = dyn_head_dense_w[:-C], dyn_head_dense_w[-C:] #(L*C+C)BSRN 
        if self.dynamic_head_seperate_param:
          dw1 = rearrange(dw1, '(L C) B S R N -> L C B S R N',C=C)
          if not self.head_dw1_norm_on_activation: dw1 = self.head_dw1_norm(dw1)
          inner_v = sum(jnp.einsum('BSND,CBSRN->CBSRD', v_outs[j], dw1[j]) for j in range(i+1)) # LBSND, (C)LBSRN->BSRD
          if self.head_dw1_norm_on_activation: inner_v = self.head_dw1_norm(inner_v) # rmsnorm
          inner_v = sum(jnp.einsum('BSND,CBSRN->CBSRD', v_outs[j], dw1[j]) for j in range(i+1)) # LBSND, (C)LBSRN->BSRD
          v_in = jnp.einsum('CBSRD,CBSRN->CBSND', inner_v, dw2)
        else:
          if not self.head_dw1_norm_on_activation: dw1 = self.head_dw1_norm(dw1)
          inner_v = sum(jnp.einsum('BSND,BSRN->BSRD', v_outs[j], dw1[j]) for j in range(i+1)) # LBSrD, (C)LBSRr->(C)BSRD # 
          if self.head_dw1_norm_on_activation: inner_v = self.head_dw1_norm(inner_v) # rmsnorm
          v_in = jnp.einsum('BSRD,CBSRN->CBSND', inner_v, dw2) # C=4, QKVO  # [LBSrD -> BSND for _ in range(4)] # CLrN
          # dynamic_conv(v_out) : LBSND, LBS-> BSND
          # dynamic_conv4(v_out) : LBSND, CLBS1-> CBSND
          # dynamic_conv4_byhead(v_out) : LBSND, CLBSN-> CBSND
          # dynamic head dense: LBSND, LBSRN -> BSRD;  BSRD,CBSRN->CBSND
          # v_in = sum(jnp.einsum('BSND,BS->BSND', v_outs[j], dw1[j,:,:,0,0]) for j in range(i+1)) ; v_in = v_in[None] # LBSND, (C)LBSRN->BSRD
          # v_in = sum(jnp.einsum('BSND,BSC->CBSND', v_outs[j], dw1[j,:,:,:,0]) for j in range(i+1))
          # v_in = sum(v_outs[j] for j in range(i+1)) / len(v_outs); v_in = v_in[None]
      x_in = x_out   
    if self.dynamic_dense_type is not None: # default 
      x_out = x_out[0] if self.dynamic_dense_fix_last_layer else x_out[1] # dynamic combination of all previous layer outputs
    elif self.hyper_conn:
      x_out = hhids.sum(axis=0)
    return x_out

  def extend_step(self,
                  inputs: JTensor,
                  *,
                  time_step: JTensor,
                  segment_pos: Optional[JTensor] = None,
                  atten_mask: Optional[JTensor] = None,
                  cross_paddings: Optional[JTensor] = None,
                  cross_segment_mask: Optional[JTensor] = None) -> JTensor:
    """Transformer stacked decoder layers, autoregressive cached decoding.

    When `inputs` has shape [B, L, D], it will do extend_step on N tokenks per
    batch. This is used to do suffix scoring after autoregressive decoding.

    When `inputs` has shape [B, D], it will do extend_step on one token per
    batch in regular autoregressive decoding.

    Args:
      inputs:         [B, D] or [B, L, D], target sequence at index time_step.
      time_step:      a 0-based scalar, the current decode step.
      segment_pos:    [B] or [B, L], the current position in the same segment.
        If unspecified, time_step will be used.
      atten_mask:     [B, 1, S] or [B, 1, L, S], optional. If None, a causal
        mask on a contiguous sequence is used by default. This is unsupported
        for cross-attention.
      cross_paddings: [B|b, S], optional 0/1 JTensor.
      cross_segment_mask: [B|b, 1, S], optional.

    Returns:
      decoder_output: [B, D], the last decoder layer output.
    """

    if self.use_cross_attention:
      assert cross_paddings is not None

    max_t = self.x_layers[0].decoding_sequence_length()
    if atten_mask is None:
      if segment_pos is None:
        segment_mask = None
      else:
        assert segment_pos.ndim == 1
        # Calculate the segment mask for this step. We assume the segment is
        # contiguous.
        # [B, T]
        src_segment_ids = jnp.where(
            jnp.arange(max_t)[jnp.newaxis, :]
            < time_step - segment_pos[:, jnp.newaxis],
            0,
            1,
        )
        # [B, 1, 1, T]
        segment_mask = attentions.segment_mask(
            jnp.ones_like(segment_pos)[:, jnp.newaxis],
            src_segment_ids,
            inputs.dtype,
        )
        # [B, 1, T]
        segment_mask = jnp.squeeze(segment_mask, 1)
      attention_mask, cross_attention_mask = (
          compute_attention_masks_for_extend_step(time_step, max_t,
                                                  segment_mask, cross_paddings,
                                                  cross_segment_mask))

    else:
      # Custom attention is assumed to have handled all padding, causality,
      # and segment masking already and can be used as is.
      attention_mask = atten_mask
      _, cross_attention_mask = compute_attention_masks_for_extend_step(
          jnp.asarray(0, dtype=jnp.uint32),
          max_t,
          None,
          cross_paddings,
          cross_segment_mask)

    decoder_input = inputs
    for layer in self.x_layers:
      decoder_output = layer.extend_step(
          decoder_input,
          time_step=time_step,
          attention_mask=attention_mask,
          segment_pos=segment_pos,
          cross_attention_mask=cross_attention_mask)
      decoder_input = decoder_output
    return decoder_output

  def transform_decode_state(
      self, transform_fn: base_layer.DecodeStateTransformFn) -> None:
    """Transforms all decode state variables based on transform_fn."""
    for layer in self.x_layers:
      layer.transform_decode_state(transform_fn)

  def lazy_broadcast_prefix(self, num_suffix_samples: int,
                            suffix_length: int) -> None:
    """Performs lazy prefix broadcast on the decoding states.

    Current decoding states will be moved to PREFIX_DECODE_CACHE. New decoding
    state will be created for the suffixes with multiple samples sharing
    previous prefixes.

    Args:
      num_suffix_samples: Number of samples that will share the same previous
        decoding state.
      suffix_length: The length of the new suffix samples.
    """
    for layer in self.x_layers:
      layer.lazy_broadcast_prefix(num_suffix_samples, suffix_length)

  def right_align_decode_state_with_prefix(
      self, max_prefix_size: int,
      right_align_fn: base_layer.DecodeStateTransformFn) -> None:
    """Right aligns decode state with prefix decode states."""
    for layer in self.x_layers:
      layer.right_align_decode_state_with_prefix(max_prefix_size,
                                                 right_align_fn)


class StackedTransformerRepeated(base_layer.BaseLayer):
  """A StackedTransformer implemented using the generic Repeat.

  Attributes:
    block: The params of a block. A block can be a single transformer
      layer,multiple layers, or a dense layer followed by a sparse layer, a.k.a.
      MOE block.
    x_times: Num times to repeat a block.
    checkpoint_policy: How to checkpoint residuals for BProp: save nothing, dot
      only or dot with no batch dimensions.
    unroll_in_decode: Whether to unroll the layers during extend_step.
    repeat_optimizer_dims_mapping: Tensor split dims mapping used for the
      optimizer state variables corresponding to the repeat prefix dims.
    nd_prefix_shape: If not None, there are multiple prefix dims of this shape
      and np.prod(nd_prefix_shape) == x_times.
  """
  block: LayerTpl = template_field(StackedTransformer)
  x_times: int = 0
  checkpoint_policy: repeats.AutodiffCheckpointType = repeats.AutodiffCheckpointType.SAVE_NOTHING
  unroll_in_decode: bool = True
  repeat_layer_name: str = 'repeat'
  sublayer_name: str = 'sub'
  repeat_optimizer_dims_mapping: SplitDimsMapping = None
  nd_prefix_shape: Optional[Sequence[int]] = None

  class WeightSharding(base_layer.BaseLayer.WeightSharding):
    """Represents how layer's learned parameters are partitioned across a mesh.

    Attributes:
      stages: How the list of blocks should be sharded.
    """
    block: SplitDimsMapping = None

  def setup(self) -> None:
    wp = self.weight_split_dims_mapping

    repeat_l_params = pax_fiddle.Config(
        repeats.Repeat,
        sub_tpl=self.block,
        x_times=self.x_times,
        checkpoint_policy=self.checkpoint_policy,
        unpack_summaries=True,
        unroll_in_decode=self.unroll_in_decode,
        sublayer_name=self.sublayer_name,
        optimizer_dims_mapping=self.repeat_optimizer_dims_mapping,
        nd_prefix_shape=self.nd_prefix_shape,
    )
    repeat_l_params.weight_split_dims_mapping.sub = wp.block

    self.create_child(self.repeat_layer_name, repeat_l_params)

  @property
  def repeat_layer(self) -> repeats.Repeat:
    return getattr(self, self.repeat_layer_name)

  def __call__(self,
               inputs: JTensor,
               paddings: JTensor,
               segment_mask: Optional[JTensor] = None,
               cross_inputs: Optional[JTensor] = None,
               cross_paddings: Optional[JTensor] = None,
               cross_segment_mask: Optional[JTensor] = None,
               segment_pos: Optional[JTensor] = None) -> JTensor:
    """Stacked Transformer layer.

    Args:
      inputs: Input sequence of shape [B, T, H].
      paddings: Input paddings of shape [B, T].
      segment_mask: Segment mask for packed input of shape [B, 1, T, T] ready to
        add to logits.
      cross_inputs: Output of the encoder, to be used for cross attention, of
        shape [B, S, H].
      cross_paddings: Paddings for cross atention of shape [B, S].
      cross_segment_mask: Segment mask for encoder-decoder in packed input case
        of shape [B, 1, T, S].
      segment_pos: Segment position of shape [B, T].

    Returns:
      Output vector with shape [B, T, D].
    """

    # TODO(zhangqiaorjc): Use positional args until nn.scan supports kwargs.
    out = self.repeat_layer(inputs, paddings, segment_mask, cross_inputs,
                            cross_paddings, cross_segment_mask, segment_pos)

    return out

  def init_states(self, *args: Any, **kwargs: Any) -> None:
    """Initialize the cache for the StackedTransformerRepeated layer.

    Args:
      *args: Other arguments.
      **kwargs: Other keyword arguments.
    Return: None.
    """
    raise NotImplementedError(type(self))

  def extend_step(self,
                  inputs: JTensor,
                  *,
                  time_step: JTensor,
                  segment_pos: Optional[JTensor] = None,
                  atten_mask: Optional[JTensor] = None,
                  cross_paddings: Optional[JTensor] = None,
                  cross_segment_mask: Optional[JTensor] = None) -> JTensor:
    """Transformer stacked decoder layers, autoregressive cached decoding.

    When `inputs` has shape [B, L, D], it will do extend_step on N tokenks per
    batch. This is used to do suffix scoring after autoregressive decoding.

    When `inputs` has shape [B, D], it will do extend_step on one token per
    batch in regular autoregressive decoding.

    Args:
      inputs: Target sequence of shape [B, D] corresponding to target sequence
        at index time_step.
      time_step: A scalar, the current decode step, 0-based.
      segment_pos: An optional JTensor of shape [B]. Current position in the
        same segment. If unspecified, time_step will be used.
      atten_mask: An optional JTensor of shape [B, 1, L, S] or [B, 1, S] for
        attention mask between inputs and the whole sequence. If it is None, it
        will be computed as a causal mask on a contiguous sequence. This passed
        in atten_mask is unsupported with cross-attention.
      cross_paddings: Source paddings - [b|B, S].
      cross_segment_mask: if not None, cross_segment_mask for this time step, of
        shape [b|B, 1, S].

    Returns:
      decoder_output: The last decoder layer output of shape [B, D].
    """

    return self.repeat_layer.extend_step(
        inputs,
        time_step=time_step,
        segment_pos=segment_pos,
        atten_mask=atten_mask,
        cross_paddings=cross_paddings,
        cross_segment_mask=cross_segment_mask)

  def transform_decode_state(
      self, transform_fn: base_layer.DecodeStateTransformFn) -> None:
    """Transforms all decode state variables based on transform_fn."""
    self.repeat_layer.transform_decode_state(transform_fn)

  def lazy_broadcast_prefix(self, num_suffix_samples: int,
                            suffix_length: int) -> None:
    """Performs lazy prefix broadcast on the decoding states.

    Current decoding states will be moved to PREFIX_DECODE_CACHE. New decoding
    state will be created for the suffixes with multiple samples sharing
    previous prefixes.

    Args:
      num_suffix_samples: Number of samples that will share the same previous
        decoding state.
      suffix_length: The length of the new suffix samples.
    """
    self.repeat_layer.lazy_broadcast_prefix(num_suffix_samples, suffix_length)

  def right_align_decode_state_with_prefix(
      self, max_prefix_size: int,
      right_align_fn: base_layer.DecodeStateTransformFn) -> None:
    """Right aligns decode state with prefix decode states."""
    self.repeat_layer.right_align_decode_state_with_prefix(
        max_prefix_size, right_align_fn)


class PipelinedTransformer(base_layer.BaseLayer):
  """A pipelined Transformer.

  Attributes:
    pipeline_stage: The layer params of each stage.
    circular_repeat: Number of round-robin layers in each stage for the circular
      pipeline schedule. If 1, this will be basic GPipe schedule.
    num_pipeline_stages: Number of pipeline stages.
    num_pipeline_microbatches: Number of pipeline microbatches.
    pipeline_microbatch_size: Size of each pipeline microbatch.
    stream_io: Whether to enable input/output streaming across stages. This is
      typically useful for DCN.
    pipeline_broadcast_inputs: If true, broadcast inputs (shared between all
      stages instead of being computed by the previous stage) will be passed
      stage-by-stage instead of being replicated.
    enable_async_circular_transfer: If True, when it is possible (which means
      num_microbatches > stages), transfers from last stage to first stage will
      be delayed in a later iteration to allow asynchronous transfers. This may
      be disabled on fast cross-stage networks to avoid extra overhead.
    bf16_accum_in_fp32: If True, use casts to make bf16 gradient accumulate in
      f32 precision.
  """
  pipeline_stage: LayerTpl = template_field(StackedTransformer)
  circular_repeat: int = 1
  num_pipeline_stages: Optional[int] = None
  num_pipeline_microbatches: Optional[int] = None
  pipeline_microbatch_size: Optional[int] = None
  stream_io: bool = False
  pipeline_broadcast_inputs: bool = False
  checkpoint_policy: AutodiffCheckpointType = AutodiffCheckpointType.SAVE_ITERATION_INPUT
  enable_async_circular_transfer: bool = True
  bf16_accum_in_fp32: bool = False

  class WeightSharding(base_layer.BaseLayer.WeightSharding):
    """Represents how layer's learned parameters are partitioned across a mesh.

    Attributes:
      stages: How the num_stages dimension should be sharded.
    """
    stages: SplitDimsMapping = (None,)

  class ActivationSharding(base_layer.BaseLayer.ActivationSharding):
    """Represents how intermediate values should be partitioned across a mesh.

    Attributes:
      final_out: How the final output should be sharded.
    """
    final_out: SplitDimsMapping = None

  def setup(self) -> None:
    assert self.num_pipeline_stages is not None and self.num_pipeline_stages > 0

    stage_params = self.pipeline_stage.clone()
    if self.circular_repeat == 1:
      pipeline_params = pax_fiddle.Config(
          pipeline.LayerwiseShardablePipelined,
          name=self.name,
          num_stages=self.num_pipeline_stages,
          single_stage_body=stage_params,
          num_microbatches=self.num_pipeline_microbatches,
          microbatch_size=self.pipeline_microbatch_size,
          unpack_summaries=True,
          stream_io=self.stream_io,
          pipeline_broadcast_inputs=self.pipeline_broadcast_inputs,
          checkpoint_policy=self.checkpoint_policy,
          bf16_accum_in_fp32=self.bf16_accum_in_fp32,
      )
    else:
      pipeline_params = pax_fiddle.Config(
          pipeline.CircularLayerwiseShardablePipelined,
          name=self.name,
          num_stages=self.num_pipeline_stages,
          circular_repeat=self.circular_repeat,
          single_stage_body=stage_params,
          num_microbatches=self.num_pipeline_microbatches,
          microbatch_size=self.pipeline_microbatch_size,
          unpack_summaries=True,
          stream_io=self.stream_io,
          pipeline_broadcast_inputs=self.pipeline_broadcast_inputs,
          checkpoint_policy=self.checkpoint_policy,
          bf16_accum_in_fp32=self.bf16_accum_in_fp32,
          enable_async_circular_transfer=self.enable_async_circular_transfer,
      )

    pipeline_params.weight_split_dims_mapping.stages = (
        self.weight_split_dims_mapping.stages
    )
    self.create_child('pipeline', pipeline_params)

  def __call__(self,
               inputs: JTensor,
               paddings: JTensor,
               segment_mask: Optional[JTensor] = None,
               cross_inputs: Optional[JTensor] = None,
               cross_paddings: Optional[JTensor] = None,
               cross_segment_mask: Optional[JTensor] = None,
               segment_pos: Optional[JTensor] = None) -> JTensor:
    """Pipelined Transformer layer.

    Args:
      inputs: Input sequence of shape [B, T, H].
      paddings: Input paddings of shape [B, T].
      segment_mask: Segment mask for packed input of shape [B, 1, T, T] ready to
        add to logits.
      cross_inputs: Output of the encoder, to be used for cross attention, of
        shape [B, S, H].
      cross_paddings: Paddings for cross atention of shape [B, S].
      cross_segment_mask: Segment mask for encoder-decoder in packed input case
        of shape [B, 1, T, S].
      segment_pos: Segment position of shape [B, T].

    Returns:
      Output vector with shape [B, T, D].
    """
    if self.pipeline_stage.cls == StackedTransformer:
      xformer_layer_p = self.pipeline_stage.transformer_layer_params_tpl
    else:
      assert self.pipeline_stage.cls == StackedTransformerRepeated
      xformer_layer_p = self.pipeline_stage.block.transformer_layer_params_tpl
    bld_mapping = xformer_layer_p.tr_atten_tpl.activation_split_dims_mapping.bld
    if not self.stream_io:
      # Annotate the inputs before the pipeline to prevent unexpected
      # propagation from earlier layers.
      inputs = base_layer.maybe_shard(inputs, bld_mapping, self.mesh_axis_names)
      if bld_mapping is not None:
        # Annotate other broadcast inputs.
        paddings = base_layer.maybe_shard(
            paddings, bld_mapping[:-1], self.mesh_axis_names
        )

        # For cross inputs, we only specify the batch dim sharding.
        def _shard_batch_dim_only(x):
          return base_layer.maybe_shard(
              x,
              [bld_mapping[0]] + [-1] * (x.ndim - 1),
              self.mesh_axis_names,
              unconstrained_dims=range(1, x.ndim),
          )

        if segment_mask is not None:
          segment_mask = _shard_batch_dim_only(segment_mask)
        if cross_inputs is not None:
          cross_inputs = _shard_batch_dim_only(cross_inputs)
        if cross_paddings is not None:
          cross_paddings = _shard_batch_dim_only(cross_paddings)
        if cross_segment_mask is not None:
          cross_segment_mask = _shard_batch_dim_only(cross_segment_mask)

        if segment_pos is not None:
          segment_pos = base_layer.maybe_shard(
              segment_pos, bld_mapping[:-1], self.mesh_axis_names
          )
    outputs = self.pipeline(
        inputs,
        paddings,
        segment_mask=segment_mask,
        cross_inputs=cross_inputs,
        cross_paddings=cross_paddings,
        cross_segment_mask=cross_segment_mask,
        segment_pos=segment_pos)

    # For non-streaming cases, we need to annotate the `outputs` twice below
    # because the first one makes sure output has the same sharding as input so
    # that the pipeine body is sharded properly.
    # The second is to switch to other shardings for later layers;
    # e.g., repurpose pipeline stages cores to data parallelism for softmax.

    if not self.stream_io:
      # Annotate the output to match input sharding.
      outputs = base_layer.maybe_shard(
          outputs, bld_mapping, self.mesh_axis_names
      )
    # Re-annotate the final output.
    outputs = base_layer.maybe_shard(
        outputs,
        self.activation_split_dims_mapping.final_out,
        self.mesh_axis_names,
    )
    return outputs

  def init_states(self, *args, **kwargs) -> NestedMap:
    raise NotImplementedError(type(self))

  def extend_step(
      self,
      inputs: JTensor,
      *,
      time_step: JTensor,
      segment_pos: Optional[JTensor] = None,
      cross_paddings: Optional[JTensor] = None,
      cross_segment_mask: Optional[JTensor] = None
  ) -> Tuple[JTensor, NestedMap]:
    raise NotImplementedError(type(self))


class PipelineCompatibleStackedTransformerRepeated(StackedTransformerRepeated):
  """Repeated transformer for inference compatible with pipeline."""

  def setup(self) -> None:
    self.repeat_layer_name = 'pipeline'
    self.sublayer_name = 'body'
    super().setup()

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

"""Implementation of the SSM-S4D layer.

S4 means Structured State Space for Sequence Modeling. See:
https://srush.github.io/annotated-s4/

S4D is its diagonal version which is simpler and performs similarly, see:
https://arxiv.org/abs/2206.11893.
"""

import jax
import math
from jax import numpy as jnp
from jax.numpy.linalg import eigh
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union
from einops import rearrange, repeat 


from praxis import base_layer
from praxis import py_utils
from praxis import pytypes
from praxis import pax_fiddle
from praxis.layers import normalizations
from praxis.layers.attentions import CausalDepthwiseConv1D

from praxis.layers import activations as activations_lib


NestedMap = py_utils.NestedMap
WeightInit = base_layer.WeightInit
WeightHParams = base_layer.WeightHParams
JTensor = pytypes.JTensor

PARAMS = base_layer.PARAMS
AUX_LOSS = base_layer.AUX_LOSS
SUMMARIES = base_layer.SUMMARIES
NON_TRAINABLE = base_layer.NON_TRAINABLE
RANDOM = base_layer.RANDOM
template_field = base_layer.template_field

LayerTpl = pax_fiddle.Config[base_layer.BaseLayer]


# RNN share weights across time dimension, so PARAMS are never split.
SCAN_SPLIT_RNGS = {PARAMS: False, RANDOM: True}


################## Hippo matrix and its normal version. #################
def make_hippo(n, hippo_type):
  """Make Hippo matrix according to https://arxiv.org/abs/2111.00396."""
  if hippo_type == "LegS":
    pmat = jnp.sqrt(1 + 2 * jnp.arange(n))
    amat = pmat[:, jnp.newaxis] * pmat[jnp.newaxis, :]
    amat = jnp.tril(amat) - jnp.diag(jnp.arange(n))
  elif hippo_type == "LagT":
    amat = jnp.ones([n, n])
    amat = jnp.tril(amat)
  return -amat


def make_nplr_hippo(n, hippo_type):
  """Make a normalized Hippo."""
  nhippo = make_hippo(n, hippo_type)  # A

  if hippo_type == "LegS":
    # Add in a rank 1 term. Makes it Normal.
    pmat = jnp.sqrt(jnp.arange(n) + 0.5)
    # HiPPO also specifies the B matrix
    bmat = jnp.sqrt(2 * jnp.arange(n) + 1.0)
  elif hippo_type == "LagT":
    # Add in a rank 1 term. Makes it Normal.
    pmat = jnp.ones(n) * jnp.sqrt(0.5)
    # HiPPO also specifies the B matrix
    bmat = jnp.ones(n)
  return nhippo, pmat, bmat


def make_dplr_hippo(n, hippo_type):
  """Diagonalized NPLR representation for S4D."""
  amat, pmat, bmat = make_nplr_hippo(n, hippo_type)  # A_diag = {-1, ..., -N}

  smat = amat + pmat[:, jnp.newaxis] * pmat[jnp.newaxis, :]  # A^N

  # Check skew symmetry
  s_diag = jnp.diagonal(smat)  # S_diag = {-0.5, ..., -0.5}, real part only
  lambda_real = jnp.mean(s_diag) * jnp.ones_like(s_diag)
  # assert jnp.allclose(Lambda_real, S_diag, atol=1e-3)

  # Diagonalize S to V \Lambda V^*
  lambda_imag, vmat = eigh(smat * -1j)

  if hippo_type == "LagT":
    lambda_imag *= float(n)  # Scaled LagT

  pmat = vmat.conj().T @ pmat
  bmat = vmat.conj().T @ bmat

  # Lambda is the same as eig(S), but eig is not supported in TPU.
  return lambda_real + 1j * lambda_imag, pmat, bmat, vmat


############### SSM kernel and layer #####################
def causal_convolution(u, ker):
  """y = conv(ker, u)."""
  assert ker.shape[-2] >= u.shape[-2], "%d, %d" % (ker.shape[-2], u.shape[-2])
  u_len = u.shape[-2]
  k_trun = ker[:u_len, :]
  ud = jnp.fft.rfft(
      jnp.float32(u), n=2*u_len, axis=-2)
  kd = jnp.fft.rfft(
      jnp.float32(k_trun), n=2*u_len, axis=-2)
  out = ud * jnp.expand_dims(kd, 0)
  return jnp.fft.irfft(out, axis=-2)[:, :u_len, :].real


def s4d_step(ab, bb, cb, u_k, x_k_1):
  """x_k = Ab x_k_1 + Bb u_k , y_k = Cb x_k."""
  # A = [nh, d], B = [nh, d], u_k = [B, d]
  x_k = ab[None, :, :] * x_k_1 + bb[None, :, :] * u_k[:, None, :]  # [B, nh, d]
  # C = [nh, d]
  y_k = jnp.sum(cb[None, :, :] * x_k, axis=1)  # [B, d]
  return x_k, y_k.real

def s4d_kernel_zoh(c, a, l, step):
  """A version of the kernel specialized to B=1 and ZOH."""
  kernel_l = c[None, :, :] * (
      jnp.exp(step[None, :, :] * a[None, :, :]) - 1) / a[None, :, :] * jnp.exp(
          l[:, None, None] * step[None, :, :] * a[None, :, :])
  return jnp.sum(kernel_l, axis=1).real  # The sum means * B where B is all-one.


def s4d_discretize(a, b, step):
  """A [nh, d] where nh represents a diagonal matrix for an input channel."""
  # A[:, i] is an nh-dim diagonal, B[:, i] is an (nh, 1) vector.
  abar = jnp.exp(step * a)
  return abar, (abar - 1) / a * b


class SSM(base_layer.BaseLayer):
  """A generic SSM layer for 1D input.

     Attributes:
       nheads: number of heads per channel.
       dim: input dimension/channel size.
       l_max: longest seq length.
       decode_num_samples: How many decoding samples for each example
       step_size: the step size for SSM discretization.
       hippo_type: which type of hippo to use.
  """
  nheads: int = 0
  dim: int = 0
  l_max: int = 0
  decode_num_samples: int = 0
  step_size: float = 0.01
  hippo_type: str = "ss4d-1d"

  def init_s4d_ac(self, nh, suffix=""):
    wp = self.weight_split_dims_mapping

    # We freeze the A matrix without training, because it behaves similarly.
    if self.hippo_type.endswith("legs"):
      amat, _, _, _ = make_dplr_hippo(nh, "LegS")
      amat = jnp.transpose(jnp.tile(amat, [self.dim, 1]), [1, 0])
    elif self.hippo_type.endswith("lagt"):
      amat, _, _, _ = make_dplr_hippo(nh, "LagT")
      amat = jnp.transpose(jnp.tile(amat, [self.dim, 1]), [1, 0])
    else:
      a_im = jnp.transpose(jnp.tile(jnp.arange(
          -jnp.pi / 2. * (nh - 1),
          jnp.pi / 2. / nh * (nh - 1) ** 2,
          jnp.pi / nh * (nh - 1)), [self.dim, 1]), [1, 0])
      a_re = jnp.ones([nh, self.dim]) * -0.5
      amat = a_re + 1j * a_im

    c_re = self.create_variable(
        "C_re" + suffix,
        WeightHParams(
            shape=[nh, self.dim],
            init=WeightInit.Gaussian(0.5 ** 0.5),
            mesh_shape=self.mesh_shape,
            tensor_split_dims_mapping=wp.wt))
    c_im = self.create_variable(
        "C_im" + suffix,
        WeightHParams(
            shape=[nh, self.dim],
            init=WeightInit.Gaussian(0.5 ** 0.5),
            mesh_shape=self.mesh_shape,
            tensor_split_dims_mapping=wp.wt))
    cmat = c_re + 1j * c_im
    return amat, cmat

  def setup(self) -> None:
    wp = self.weight_split_dims_mapping

    # We freeze the step_size without training, because it behaves better.
    self.ss = jnp.ones([1, self.dim]) * self.step_size
    self.dmat = self.create_variable(
        "D",
        WeightHParams(
            shape=[1, self.dim],
            init=WeightInit.Constant(1.),
            mesh_shape=self.mesh_shape,
            tensor_split_dims_mapping=wp.wt))

    l = jnp.arange(self.l_max)
    self.amat, self.cmat = self.init_s4d_ac(self.nheads)  # [nh, d]
    self.ssm_k = s4d_kernel_zoh(self.cmat, self.amat, l, self.ss)  # [L, d]
    self.abar, self.bbar = s4d_discretize(
        self.amat,
        jnp.ones((self.nheads, self.dim)), self.ss)  # B is just ones.

  def init_states(self, batch_size: int) -> None:
    """Initialize the ssm state to all zeros."""
    state = jnp.zeros((batch_size, self.nheads, self.dim), dtype=jnp.complex64)
    self.update_decode_state("ssm_state", state)

  def extend_step(self,
                  inputs: JTensor) -> JTensor:
    """Extends SSM for one step on 'inputs' from the last 'ssm_state'.

    Args:
      inputs: A JTensor of inputs.

    Returns:
      outputs: A JTensor
    """
    batch_size = inputs.shape[0]
    if not self.has_variable(base_layer.DECODE_CACHE, "ssm_state"):
      self.init_states(batch_size)

    x = self.get_decode_state("ssm_state")
    assert x.shape[0] == batch_size, (x.shape[0], batch_size)

    x, y = s4d_step(self.abar, self.bbar, self.cmat, inputs, x)
    self.update_decode_state("ssm_state", x)
    return y + self.dmat * inputs

  def __call__(self, inputs: JTensor) -> JTensor:
    """Use convolution to compute outputs in parallel.

    Args:
      inputs: A JTensor of inputs.

    Returns:
      outputs: A JTensor
    """
    batch_size = inputs.shape[0]
    # the batch is x decode_num_samples in extend_step decoding.
    self.init_states(batch_size * self.decode_num_samples)

    y = causal_convolution(inputs, self.ssm_k)
    return y + self.dmat[None, :, :] * inputs


class Mamba2(base_layer.BaseLayer): # mqy
  '''
  Implement mamba2_simple in jax.
  Ref: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba2_simple.py#L24
  config of Mamba2-370m
  {
    "d_model": 1024,
    "d_intermediate": 0,
    "n_layer": 48,
    "vocab_size": 50277,
    "ssm_cfg": {
        "layer": "Mamba2"
    },
    "attn_layer_idx": [],
    "attn_cfg": {},
    "rms_norm": true,
    "residual_in_fp32": true,
    "fused_add_norm": true,
    "pad_vocab_size_multiple": 16,
    "tie_embeddings": true
  }
  '''
  d_model: int = 1024
  d_state: int = 128
  d_conv: int = 4
  expand: int = 2
  headdim: int = 64
  ngroups: int = 1
  A_init_range: Tuple = (1, 16)
  dt_min: float = 0.001
  dt_max: float = 0.1
  dt_init_floor: float = 1e-4
  dt_limit: Tuple = (0.0, float("inf"))
  bias: bool = False # bias for linear transformation
  conv_bias: bool = True
  chunk_size: int = 256
  layer_idx: Optional[int] = None
  learnable_init_states: bool = False
  activation_cls: activations_lib.BaseActivation = activations_lib.SiLU
  use_mem_eff_path: bool = False
  norm_tpl: LayerTpl = template_field(normalizations.RmsNorm) 
  skip_weight_decay: bool = True
  seed: int = 124 # change seed in different layers: seed + layer_idx 
  use_minimal: bool = True

  def setup(self) -> None:
    self.d_inner = self.expand * self.d_model
    assert self.d_inner % self.headdim == 0
    self.nheads = self.d_inner // self.headdim

    # Order: [z: d_inner, x: d_inner, B: d_state, C: d_state, dt: nheads] 
    d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
    pc = WeightHParams(
          shape=[self.d_model, d_in_proj], # DX
          mesh_shape=self.mesh_shape, tensor_split_dims_mapping=['data', None],
          init=WeightInit.Gaussian(math.sqrt(2.0 / (self.d_model + d_in_proj))))
    self.create_variable('in_proj', pc)

    conv_dim = self.d_inner + 2 * self.ngroups * self.d_state

    self.create_child('conv1d', pax_fiddle.Config(CausalDepthwiseConv1D).clone().set(hidden_dims=conv_dim, kernel_size=self.d_conv, bias=self.conv_bias, skip_weight_decay=False))

    # if self.learnable_init_states:
    #   self.init_states = nn.Parameter(torch.zeros(self.nheads, self.headdim, self.d_state, **factory_kwargs))
    #   self.init_states._no_weight_decay = True

    self.create_child('act', pax_fiddle.Config(self.activation_cls).clone())

    # prng_key = jax.random.PRNGKey(seed=self.seed+self.layer_idx)
    # prng_key_dt, prng_key_a = jax.random.split(prng_key, num=2)

    # TODO:
    # prng_key_dt = jnp.array([1896456402,   17229315, 4, 3], dtype=jnp.uint32)
    # prng_key_a = jnp.array([4081828428, 1707601653, 7, 9], dtype=jnp.uint32)

    # init_dtype = jnp.float32
    # Initialize log dt bias
    # dt = jnp.exp(
    #      jax.random.uniform(prng_key_dt, [self.nheads], minval=0, maxval=1.0) * (math.log(self.dt_max) - math.log(self.dt_min))
    #     #  jnp.arange(1,17,1)/16 * (math.log(self.dt_max) - math.log(self.dt_min))
    #     + math.log(self.dt_min)
    # )
    # dt = dt.clip(min=self.dt_init_floor)
    # # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
    # inv_dt = dt + jnp.log(-jnp.expm1(-dt))
    dt_bias = WeightHParams(
          shape=[self.nheads], # N
          mesh_shape=self.mesh_shape, tensor_split_dims_mapping=[None],
          # init=WeightInit.Constant(inv_dt),
          init=WeightInit.MambaDtBias(1),
          collections=None if not self.skip_weight_decay else [
                base_layer.WeightHParamsCollection.SKIP_LP_REGULARIZATION 
            ]
          )
    self.create_variable('dt_bias', dt_bias) # nheads

    assert self.A_init_range[0] > 0 and self.A_init_range[1] >= self.A_init_range[0]
    # A = jax.random.uniform(prng_key_a, [self.nheads], minval=self.A_init_range[0], maxval=self.A_init_range[1]) # [1,16]
    # A_log_init = jnp.log(A)
    A_log = WeightHParams(
          shape=[self.nheads], # N
          mesh_shape=self.mesh_shape, tensor_split_dims_mapping=[None],
          # init=WeightInit.Constant(A_log_init),
          init=WeightInit.MambaALog(1),
          collections=None if not self.skip_weight_decay else [
                base_layer.WeightHParamsCollection.SKIP_LP_REGULARIZATION 
            ])
    self.create_variable('A_log', A_log) # nheads

    D = WeightHParams(
          shape=[self.nheads], # N
          mesh_shape=self.mesh_shape, tensor_split_dims_mapping=[None],
          init=WeightInit.Constant(1),
          collections=None if not self.skip_weight_decay else [
                base_layer.WeightHParamsCollection.SKIP_LP_REGULARIZATION 
            ])
    self.create_variable('D', D) # nheads

    self.create_child('norm', self.norm_tpl.clone().set(name='rmsnorm', dim=self.d_inner, epsilon=1e-5))
                      
    pc = WeightHParams(
          shape=[self.d_inner, self.d_model], # DX
          mesh_shape=self.mesh_shape, tensor_split_dims_mapping=[None, 'data'],
          init=WeightInit.Gaussian(math.sqrt(2.0 / (self.d_model + self.d_inner))))
    self.create_variable('out_proj', pc)                  
    return 
  
  def __call__(self, u: JTensor, seq_idx=None) -> JTensor:
    batch, seqlen, dim = u.shape
    zxbcdt = jnp.einsum('b l d, d i -> b l i' , u, self.theta.in_proj)
    A = -jnp.exp(self.theta.A_log) # initialized in range [1, 16] 
    # initial_states = None
    # dt_limit_kwargs = {} 

    # split self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads
    idx_z, idx_xBC = [self.d_inner, self.d_inner*2 + 2 * self.ngroups * self.d_state]
    z, xBC, dt = zxbcdt[..., :idx_z], zxbcdt[..., idx_z:idx_xBC], zxbcdt[..., idx_xBC:] # z: gate, xBC: VKQ, dt: dynamic a 
    dt = jax.nn.softplus(dt + self.theta.dt_bias)  # (B, L, nheads) dynamic dt + static dt, initialized in range [0.001, 0.1]

    # dconv
    # xBC = self.act(self.conv1d(xBC.swapaxes(1, 2), -1)).swapaxes(1, 2)  # (B, L, self.d_inner + 2 * ngroups * d_state)
    xBC = self.act(self.conv1d(xBC, 1))  # (B, L, self.d_inner + 2 * ngroups * d_state)
    # split 
    x, B, C = xBC[..., :self.d_inner], xBC[..., self.d_inner: self.d_inner + self.ngroups * self.d_state], xBC[..., self.d_inner + self.ngroups * self.d_state:]
    
    if self.use_minimal:
      initial_states = jnp.zeros((batch, 1, self.d_inner // self.headdim, self.headdim, self.d_state)) # b1hpn
      x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
      X = x * dt[...,None] # blhp,blh1 -> blhp 
      A = A * dt # h * blh; static * dynamic -> dynamic
      B = rearrange(B, "b l (g n) -> b l g n", g=self.ngroups)
      C = rearrange(C, "b l (g n) -> b l g n", g=self.ngroups)
      y, _ = ssd_minimal_discrete(X, A, B, C, self.chunk_size, initial_states=initial_states)
      y = y + x * rearrange(self.theta.D, "h -> h 1")
    else:
      y = mamba_chunk_scan(
          rearrange(x, "b l (h p) -> b l h p", p=self.headdim), # V: B, L, H, head_dim, split V into H heads
          dt, # B,L,nheads
          A, # nheads
          rearrange(B, "b l (g n) -> b l g n", g=self.ngroups), # K: B, L, head_dim
          rearrange(C, "b l (g n) -> b l g n", g=self.ngroups), # Q: B, L, head_dim
          self.chunk_size, # 256
          D=self.theta.D, # nheads 
          z=None,
          # seq_idx=seq_idx,
          # initial_states=initial_states,
          # **dt_limit_kwargs,
      )

    y = rearrange(y, "b l h p -> b l (h p)") # B, L, model_dim
    y = self.norm(y * jax.nn.silu(z))  # new normalization 
    out = jnp.einsum('b l i, i d -> b l d' , y, self.theta.out_proj)

    return out

def ssd_minimal_discrete(X, A, B, C, block_len, initial_states=None):
    """
    Arguments:
        X: (batch, length, n_heads, d_head)
        A: (batch, length, n_heads)
        B: (batch, length, n_heads, d_state)
        C: (batch, length, n_heads, d_state)
    Return:
        Y: (batch, length, n_heads, d_head)
    """
    assert X.dtype == A.dtype == B.dtype == C.dtype
    assert X.shape[1] % block_len == 0

    # Rearrange into blocks/chunks
    X, A, B, C = [rearrange(x, "b (c l) ... -> b c l ...", l=block_len) for x in (X, A, B, C)]

    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = jnp.cumsum(A, axis=-1) # bhcl

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    L = jnp.exp(segsum(A)) # b h c l l
    Y_diag  = jnp.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)

    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    decay_states = jnp.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
    states = jnp.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X) # apply inner-chunk decay 

    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
    # (middle term of factorization of off-diag blocks; A terms)
    # if initial_states is None:
    #     initial_states = jnp.zeros_like(states[:, :1])
    states = jnp.concatenate([initial_states, states], axis=1)
    decay_chunk = jnp.exp(segsum(jnp.pad(A_cumsum[:, :, :, -1], ((0,0),(0,0),(1, 0))))) # chunk_level decay:  b h c -> b h c c 1 1 
    new_states = jnp.einsum("bhzc,bchpn->bzhpn", decay_chunk, states) # bchpn apply chunk_level decay
    states, final_state = new_states[:, :-1], new_states[:, -1] # previous chunks, last chunk 

    # 4. Compute state -> output conversion per chunk
    # (left term of low-rank factorization of off-diagonal blocks; C terms)
    state_decay_out = jnp.exp(A_cumsum)
    Y_off = jnp.einsum('bclhn,bchpn,bhcl->bclhp', C, states, state_decay_out) # query 

    # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
    Y = rearrange(Y_diag+Y_off, "b c l h p -> b (c l) h p")
    return Y, final_state



def mamba_chunk_scan(x, dt, A, B, C, chunk_size, D=None, z=None, dt_bias=None, dt_softplus=False)-> JTensor:
  """
  Argument:
      x: (batch, seqlen, nheads, headdim)
      dt: (batch, seqlen, nheads)
      A: (nheads)
      B: (batch, seqlen, ngroups, dstate)
      C: (batch, seqlen, ngroups, dstate)
      D: (nheads, headdim) or (nheads,)
      z: (batch, seqlen, nheads, headdim)
      dt_bias: (nheads,)
  Return:
      out: (batch, seqlen, nheads, headdim)
  """
  batch, seqlen, nheads, headdim = x.shape
  dstate = B.shape[-1]
  assert seqlen % chunk_size == 0
  # if seqlen % chunk_size != 0:
  #     dt = F.pad(dt, (0, 0, 0, chunk_size - seqlen % chunk_size))
  dt = rearrange(dt, "b (c l) h -> b h c l", l=chunk_size)
  dt = dt.astype(jnp.float32)  # We want high precision for this before cumsum
  # if dt_bias is not None:
  #     dt = dt + rearrange(dt_bias, "h -> h 1 1")
  # if dt_softplus:
  #     dt = F.softplus(dt)
  dA = dt * rearrange(A, "h -> h 1 1") # bhcl * h11 -> bhcl ; dynamic_dt * static_A 
  dA_cumsum = jnp.cumsum(dA, axis=-1)
  # 1. Compute the state for each chunk
  states = chunk_state(B, x, dt, dA_cumsum)
  # 2. Pass the state to all the chunks by weighted cumsum.
  states = rearrange(state_passing(rearrange(states, "... p n -> ... (p n)"), dA_cumsum[:, :, :, -1])[0],
                      "... (p n) -> ... p n", n=dstate)
  # 3. Compute the output for each chunk
  out = chunk_scan(B, C, x, dt, dA_cumsum, states, D=D, z=z)
  return out 

def chunk_state(B, x, dt, dA_cumsum):
  """
  Argument:
      B: (batch, seqlen, ngroups, headdim)
      x: (batch, seqlen, nheads, headdim)
      dt: (batch, nheads, nchunks, chunk_size)
      dA_cumsum: (batch, nheads, nchunks, chunk_size)
  Return:
      states: (batch, nchunks, nheads, headdim, dstate)
  """
  # Check constraints.
  batch, seqlen, nheads, headdim = x.shape
  dstate = B.shape[-1]
  _, _, nchunks, chunk_size = dt.shape
  assert seqlen <= nchunks * chunk_size
  assert x.shape == (batch, seqlen, nheads, headdim)
  assert dt.shape == (batch, nheads, nchunks, chunk_size)
  ngroups = B.shape[2]
  assert nheads % ngroups == 0
  assert B.shape == (batch, seqlen, ngroups, dstate)
  B = repeat(B, "b l g d -> b l (g h) d", h=nheads // ngroups) # expand shared K: (B, L, H, head_dim)
  assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
  assert seqlen == nchunks * chunk_size
  # if seqlen < nchunks * chunk_size:
  #     x = F.pad(x, (0, 0, 0, 0, 0, nchunks * chunk_size - seqlen))
  #     B = F.pad(B, (0, 0, 0, 0, 0, nchunks * chunk_size - seqlen))
  x = rearrange(x, "b (c l) h p -> b c l h p", l=chunk_size) # V: B, nchunks, chunk_size, nheads, head_dim
  B = rearrange(B, "b (c l) ... -> b c l ...", l=chunk_size) # K: B, nchunks, chunk_size, nheads, head_dim
  # TODO: check decay_states
  decay_states = jnp.exp((dA_cumsum[:, :, :, -1:] - dA_cumsum)) # L in the paper: exp(log(cumprod(A))) 
  return jnp.einsum("bclhn,bhcl,bhcl,bclhp->bchpn", B.astype(x.dtype), decay_states.astype(x.dtype), dt.astype(x.dtype), x)

def state_passing(states, dA_chunk_cumsum, initial_states=None):
  """
  Argument:
      states: (batch, nchunks, nheads, dim)
      dA_chunk_cumsum: (batch, nheads, nchunks)
      initial_states: (batch, nheads, dim)
  Return:
      out: (batch, nchunks, nheads, dim)
      final_states: (batch, nheads, dim)
  """
  if initial_states is None:
      initial_states = jnp.zeros_like(states[:, 0]) # bhd
  states = jnp.concatenate([rearrange(initial_states, "b h d -> b 1 h d"), states], axis=1) # b(1+c)hd
  dA_chunk_cumsum = jnp.pad(dA_chunk_cumsum, ((0,0),(0,0),(1, 0))) # bhc->bh(1+c)
  dA_chunk_cumsum = jnp.cumsum(dA_chunk_cumsum, axis=-1) # bh(1+c)
  nchunks = dA_chunk_cumsum.shape[-1]
  # (batch, nheads, nchunks, nchunks)
  dt_chunk_segment_sum = dA_chunk_cumsum[:, :, :, None] - dA_chunk_cumsum[:, :, None, :] # bh(1+c)1 - bh1(1+c) -> bh(1+c)(1+c)
  # (batch, nheads, nchunks, nchunks)
  decay_chunk = jnp.exp(dt_chunk_segment_sum) #bh(1+c)(1+c)
  causal_mask = jnp.tril(jnp.ones((nchunks, nchunks), dtype=bool), k=0) # (1+c)(1+c)
  # decay_chunk = decay_chunk.masked_fill(~causal_mask, 0)
  decay_chunk = jnp.where(causal_mask[None,None], decay_chunk, 0)
  out = jnp.einsum("bhzc,bchd->bzhd", decay_chunk.astype(states.dtype), states)
  return out[:, :-1], out[:, -1] # bchd, bhd

def chunk_scan(B, C, x, dt, dA_cumsum, prev_states, D=None, z=None):
  """
  Argument:
      B: (batch, seqlen, ngroups, dstate)
      C: (batch, seqlen, ngroups, dstate)
      x: (batch, seqlen, nheads, headdim)
      dt: (batch, nheads, nchunks, chunk_size)
      dA_cumsum: (batch, nheads, nchunks, chunk_size)
      prev_states: (batch, nchunks, nheads, headdim, dstate)
      D: (nheads, headdim) or (nheads,)
      z: (batch, seqlen, nheads, headdim)
  Return:
      out: (batch, seqlen, nheads, headdim)
  """
  batch, seqlen, nheads, headdim = x.shape
  _, _, ngroups, dstate = B.shape
  assert B.shape == (batch, seqlen, ngroups, dstate)
  _, _, nchunks, chunk_size = dt.shape
  assert seqlen == nchunks * chunk_size
  assert C.shape == B.shape, f'BC:{B.shape} and {C.shape}'
  B = repeat(B, "b l g d -> b l (g h) d", h=nheads // ngroups)
  C = repeat(C, "b l g d -> b l (g h) d", h=nheads // ngroups)
  CB = jnp.einsum("bclhn,bcshn->bchls", rearrange(C, "b (c l) h n -> b c l h n", c=nchunks),
                    rearrange(B, "b (c s) h n -> b c s h n", c=nchunks)) # bchls
  # (batch, nheads, nchunks, chunksize, chunksize)
  dt_segment_sum = dA_cumsum[:, :, :, :, None] - dA_cumsum[:, :, :, None, :]# bhcl1 - bhc1l -> bhcll
  decay = jnp.exp(dt_segment_sum) # bhcll
  scores_decay = CB * rearrange(decay, "b h c l s -> b c h l s") # bchls
  causal_mask = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=bool), k=0)
  scores_decay = jnp.where(causal_mask[None,None,None], scores_decay, 0) # bchls
  # scores_decay = scores_decay.masked_fill(~causal_mask, 0)
  out = jnp.einsum('bchls,bhcs,bcshp->bclhp', scores_decay.astype(x.dtype), dt.astype(x.dtype),
                      rearrange(x, "b (c s) h p -> b c s h p", c=nchunks))
  state_decay_out = jnp.exp(rearrange(dA_cumsum, "b h c l -> b c l h 1"))
  out_prev = jnp.einsum('bclhn,bchpn->bclhp', rearrange(C, "b (c l) h n -> b c l h n", c=nchunks),
                          prev_states.astype(C.dtype)) * state_decay_out
  out = out + out_prev
  out = rearrange(out, "b c l h p -> b (c l) h p")
  if D is not None:
      if len(D.shape) == 1:
          D = rearrange(D, "h -> h 1")
      out = out + x * D # blhd + blhd * (h1)
  return out if z is None else out * jax.nn.silu(z)

def segsum(x):
    """More stable segment sum calculation."""
    T = x.shape[-1]
    x = repeat(x, "... d -> ... d e", e=T)
    mask = jnp.tril(jnp.ones((T, T), dtype=bool), k=-1)
    x = jnp.where(mask, x, 0)
    # x = x.masked_fill(~mask, 0)
    x_segsum = jnp.cumsum(x, axis=-2)
    mask = jnp.tril(jnp.ones((T, T), dtype=bool), k=0)
    # x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    x_segsum = jnp.where(mask, x_segsum, -jnp.inf)
    return x_segsum
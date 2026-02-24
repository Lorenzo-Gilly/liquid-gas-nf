#!/usr/bin/python
#
# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Distrax bijectors."""

from typing import Union

import chex
import distrax
import jax.numpy as jnp

Array = chex.Array


class CircularShift(distrax.Lambda):
  """Shift with wrapping around. Or, translation on a torus."""

  def __init__(self,
               shift: Array,
               lower: Union[float, Array],
               upper: Union[float, Array]):
    # Checking `upper >= lower` causes jitting problems when `upper` and
    # `lower` are arrays, we check only for scalars. The check is not
    # really necessary, since wrapping works equally well when upper < lower.
    if jnp.isscalar(lower) and jnp.isscalar(upper) and (lower >= upper):
      raise ValueError('`lower` must be less than `upper`.')

    try:
      width = upper - lower
    except TypeError as e:
      raise ValueError('`lower` and `upper` must be broadcastable to same '
                       f'shape, but `lower`={lower} and `upper`={upper}') from e

    wrap = lambda x: jnp.mod(x - lower, width) + lower
    # We first wrap the shift to avoid `x + shift` when `shift` is very large,
    # which can lead to loss of precision. This gives the same result, since
    # `wrap(x + wrap(shift)) == wrap(x + shift)`. Same holds for `y - shift`.
    shift = wrap(shift)
    super().__init__(
        forward=lambda x: wrap(x + shift),
        inverse=lambda y: wrap(y - shift),
        forward_log_det_jacobian=jnp.zeros_like,
        inverse_log_det_jacobian=jnp.zeros_like,
        event_ndims_in=0,
        is_constant_jacobian=True)


class RemoveOrigin(distrax.Bijector):
  """Removes/adds particle 0 (the origin particle) from the event.

  Forward: [..., N, D] -> [..., (N-1)*D]  (strip particle 0, flatten)
  Inverse: [..., (N-1)*D] -> [..., N, D]  (prepend zeros, reshape)
  Log-det is 0 in both directions (no volume change).
  """

  def __init__(self, n_particles: int, dimensions: int):
    super().__init__(event_ndims_in=2, event_ndims_out=1)
    self._n_particles = n_particles
    self._dimensions = dimensions

  def forward_and_log_det(self, x: Array):
    # x: [..., N, D] -> [..., (N-1)*D]
    stripped = x[..., 1:, :]  # remove particle 0
    flat = stripped.reshape(x.shape[:-2] + ((self._n_particles - 1) * self._dimensions,))
    return flat, jnp.zeros(x.shape[:-2])

  def inverse_and_log_det(self, y: Array):
    # y: [..., (N-1)*D] -> [..., N, D]
    batch_shape = y.shape[:-1]
    reshaped = y.reshape(batch_shape + (self._n_particles - 1, self._dimensions))
    zeros = jnp.zeros(batch_shape + (1, self._dimensions))
    full = jnp.concatenate([zeros, reshaped], axis=-2)
    return full, jnp.zeros(batch_shape)


class Rescale(distrax.ScalarAffine):
  """Rescales from the range [lower_in, upper_in] to [lower_out, upper_out]."""

  def __init__(self,
               lower_in: Union[float, Array],
               upper_in: Union[float, Array],
               lower_out: Union[float, Array],
               upper_out: Union[float, Array]):
    width_in = upper_in - lower_in
    width_out = upper_out - lower_out
    scale = width_out / width_in
    super().__init__(scale=scale, shift=lower_out - scale * lower_in)

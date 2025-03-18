from typing import Tuple

import chex
import jax
import jax.numpy as jnp


class Box:
	"""
	Minimal jittable class for array-shaped gymnax spaces.
	"""
	def __init__(
		self,
		low: float,
		high: float,
		shape: Tuple[int],
		dtype: jnp.dtype = jnp.float32,
	):
		self.low = low
		self.high = high
		self.shape = shape
		self.dtype = dtype

	def sample(self, rng: chex.PRNGKey) -> chex.Array:
		"""Sample random action uniformly from 1D continuous range."""
		return jax.random.uniform(
			rng, shape=self.shape, minval=self.low, maxval=self.high
		).astype(self.dtype)
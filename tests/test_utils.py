import jax
import jax.numpy as jnp

from camar.utils import Box


def test_box_sample():
    key = jax.random.key(0)
    low = -5.0
    high = 5.0
    shape = (4,)
    dtype = jnp.float32
    box_space = Box(low=low, high=high, shape=shape, dtype=dtype)

    sample = box_space.sample(key)

    assert sample.shape == shape
    assert jnp.isdtype(sample.dtype, dtype)
    assert jnp.all(sample >= low)
    assert jnp.all(sample < high)

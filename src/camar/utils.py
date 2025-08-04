import jax
import jax.numpy as jnp
from flax import struct
from jax import Array
from jax.typing import ArrayLike

from .dynamics import PhysicalState


class Box:
    def __init__(
        self,
        low: float,
        high: float,
        shape: tuple | int,
        dtype: jnp.dtype = jnp.float32,
    ):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype

    def sample(self, rng: ArrayLike) -> Array:
        """Sample random action uniformly from 1D continuous range."""
        return jax.random.uniform(
            rng,
            shape=self.shape,
            minval=self.low,
            maxval=self.high,
            dtype=self.dtype,
        )


@struct.dataclass
class State:
    physical_state: PhysicalState

    landmark_pos: ArrayLike  # [num_landmarks, 2]

    goal_pos: ArrayLike  # [num_agents, [x, y]]

    sizes: "Sizes"  # noqa: F821 see maps/base.py

    is_collision: ArrayLike  # [num_agents, ]

    # done: ArrayLike  # bool [num_agents, ]
    step: ArrayLike | int  # current step

    # metrics
    on_goal: ArrayLike  # [num_agents, ]
    time_to_reach_goal: ArrayLike  # [num_agents, ]
    num_collisions: ArrayLike  # [num_agents, ]

    goal_keys: ArrayLike  # [num_agents, ] or [] - jax keys for the controllable goal generation (keys are updated only for agents on_goal in lifelong)

from typing import Tuple

from jax import Array
from jax.typing import ArrayLike

import jax
import jax.numpy as jnp
from flax import struct


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

    def sample(self, rng: ArrayLike) -> Array:
        """Sample random action uniformly from 1D continuous range."""
        return jax.random.uniform(
            rng, shape=self.shape, minval=self.low, maxval=self.high
        ).astype(self.dtype)


@struct.dataclass
class State:
    agent_pos: ArrayLike  # [num_entities, [x, y]]
    agent_vel: ArrayLike  # [n, [x, y]]

    goal_pos: ArrayLike  # [num_agents, [x, y]]
    # obstacle_pos: ArrayLike  # [num_obstacles, [x, y]]
    landmark_pos: ArrayLike  # [num_landmarks, [x, y]]

    # observation: ArrayLike  # [num_agents, max_obs, 2]
    # reward: ArrayLike  # [num_agents]

    is_collision: ArrayLike # [num_agents, ]

    # done: ArrayLike  # bool [num_agents, ]
    step: int  # current step
    # metrics
    on_goal: ArrayLike  # [num_agents, ]
    time_to_reach_goal: ArrayLike  # [num_agents, ]
    num_collisions: ArrayLike # [num_agents, ]
    # flowtime: float # current flowtime
    # makespan: float # current makespan

    goal_keys: ArrayLike  # [num_agents, ] or [] - jax keys for the controllable goal generation (keys are updated only for agents on_goal in lifelong)

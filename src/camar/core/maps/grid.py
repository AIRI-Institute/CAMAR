from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from camar.core.maps.base import Map


def convert_map_str(map_str, remove_border, add_border):
    map_array = jnp.array([[0 if char == "." else 1 for char in line] for line in map_str.split("\n") if line])

    if remove_border:
        map_array = map_array[1:-1, 1:-1]

    if add_border:
        map_array = jnp.pad(map_array, pad_width=[(1, 1), (1, 1)], constant_values=[(1, 1), (1, 1)])

    return map_array


class Grid(Map):
    """fixed obstacles' positions, random agents' positions"""
    def __init__(
        self,
        map_str: str,
        num_agents: int = 10,
        obstacle_size: float = 0.1,
        agent_size: float = 0.09,
        remove_border: bool = False,
        add_border: bool = True,
    ) -> Map:
        self.num_agents = num_agents
        self.obstacle_size = obstacle_size
        self.agent_size = agent_size

        map_array = convert_map_str(map_str, remove_border, add_border)

        num_rows, num_cols = map_array.shape

        map_idx_rows, map_idx_cols = jnp.meshgrid(jnp.arange(num_cols), jnp.arange(num_rows))
        self.map_idx = jnp.stack((map_idx_cols, map_idx_rows), axis=2).reshape(-1, 2) # for random agents' positions
        self.is_free = 1 - map_array.flatten()

        self.height = num_rows * obstacle_size
        self.width = num_cols * obstacle_size

        landmark_idx_x, landmark_idx_y = jnp.nonzero(map_array)

        landmark_x, landmark_y = self.idx2coords(landmark_idx_x, landmark_idx_y)

        # landmark_x = landmark_x * obstacle_size - self.height / 2 + obstacle_size / 2
        # landmark_y = landmark_y * obstacle_size - self.width / 2 + obstacle_size / 2

        self.landmark_pos = jnp.stack((landmark_y, landmark_x), axis=1)

        self.num_landmarks = self.landmark_pos.shape[0]

    @property
    def landmark_rad(self) -> float:
        return self.obstacle_size / 2

    @property
    def agent_rad(self):
        return self.agent_size / 2

    @property
    def goal_rad(self):
        return self.agent_rad / 2.5

    @partial(jax.jit, static_argnums=[0])
    def reset(self, key: ArrayLike) -> Tuple[Array, Array, Array, Array]:

        key_agents, key_goals = jax.random.split(key, 2)

        agent_pos = self.generate_agent_pos(key_agents)

        # generate goals
        goal_idx = jax.random.choice(key_goals, self.map_idx, p=self.is_free / self.is_free.sum(), replace=False, shape=(self.num_agents, ))

        goal_x, goal_y = self.idx2coords(goal_idx[:, 0], goal_idx[:, 1])
        goal_pos = jnp.stack((goal_y, goal_x), axis=-1)

        return key_goals, self.landmark_pos, agent_pos, goal_pos

    @partial(jax.jit, static_argnums=[0])
    def reset_lifelong(self, key) -> Tuple[Array, Array, Array, Array]:
        key_agents, key_goals = jax.random.split(key, 2)

        agent_pos = self.generate_agent_pos(key_agents)

        # key for each agent
        key_goals = jax.random.split(key_goals, self.num_agents)

        goal_pos = self.generate_goal_pos(key_goals)

        return key_goals, self.landmark_pos, agent_pos, goal_pos

    def update_goals(self, keys: ArrayLike, goal_pos: ArrayLike, to_update: ArrayLike) -> Tuple[Array, Array]:
        new_keys = jax.vmap(jax.random.split, in_axes=[0, None])(keys, 1)[:, 0]
        new_keys = jnp.where(to_update, new_keys, keys)

        new_goal_pos = self.generate_goal_pos(new_keys)

        return new_keys, new_goal_pos

    @partial(jax.vmap, in_axes=[None, 0])
    def generate_goal_pos(self, key: ArrayLike) -> Array:
        goal_idx = jax.random.choice(key, self.map_idx, p=self.is_free / self.is_free.sum(), replace=False)

        goal_x, goal_y = self.idx2coords(goal_idx[0], goal_idx[1])
        goal_pos = jnp.stack((goal_y, goal_x), axis=-1)
        return goal_pos

    @partial(jax.jit, static_argnums=[0])
    def generate_agent_pos(self, key: ArrayLike) -> Array:

        agent_idx = jax.random.choice(key, self.map_idx, p=self.is_free, replace=False, shape=(self.num_agents, )) # TODO: it will work only if agent_size <= obstacle_size
        agent_x, agent_y = self.idx2coords(agent_idx[:, 0], agent_idx[:, 1])
        agent_pos = jnp.stack((agent_y, agent_x), axis=-1)

        return agent_pos

    def idx2coords(self, idx_x, idx_y):
        coord_x = idx_x * self.obstacle_size - self.height / 2 + self.obstacle_size / 2
        coord_y = idx_y * self.obstacle_size - self.width / 2 + self.obstacle_size / 2

        return coord_x, coord_y

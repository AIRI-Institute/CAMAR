from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from camar.maps.base import BaseMap


def map_str2array(map_str, remove_border, add_border):
    map_array = jnp.array([[1 if char in set("@*#") else 0 for char in line] for line in map_str.split("\n") if line])

    if remove_border:
        map_array = map_array[1:-1, 1:-1]

    if add_border:
        map_array = jnp.pad(map_array, pad_width=[(1, 1), (1, 1)], constant_values=[(1, 1), (1, 1)])

    return map_array

def idx2pos(idx_x, idx_y, obstacle_size, height, width):
    coord_x = idx_x * obstacle_size - height / 2 + obstacle_size / 2
    coord_y = idx_y * obstacle_size - width / 2 + obstacle_size / 2

    return jnp.stack((coord_y, coord_x), axis=1)

def parse_map_array(map_array, obstacle_size):
    num_rows, num_cols = map_array.shape

    map_idx_rows, map_idx_cols = jnp.meshgrid(jnp.arange(num_cols), jnp.arange(num_rows))

    height = num_rows * obstacle_size
    width = num_cols * obstacle_size

    # obstacles
    landmark_idx_x, landmark_idx_y = jnp.nonzero(map_array)
    landmark_pos = idx2pos(landmark_idx_x, landmark_idx_y, obstacle_size, height, width)

    # free cells
    map_idx = jnp.stack((map_idx_cols, map_idx_rows), axis=2).reshape(-1, 2) # for random agent and goal positions
    is_free = ~map_array.flatten().astype(jnp.bool_)
    free_idx = map_idx[is_free, :]
    free_pos = idx2pos(free_idx[:, 0], free_idx[:, 1], obstacle_size, height, width)

    return landmark_pos, free_pos, height, width


class StringGrid(BaseMap):
    def __init__(
        self,
        map_str: str,
        agent_idx: Optional[ArrayLike] = None,
        goal_idx: Optional[ArrayLike] = None,
        num_agents: Optional[int] = 10,
        random_agents: Optional[bool] = True,
        random_goals: Optional[bool] = True,
        remove_border: bool = False,
        add_border: bool = True,
        obstacle_size: float = 0.1,
        agent_size: float = 0.04,
    ) -> BaseMap:
        if agent_idx is not None:
            num_agents = agent_idx.shape[0]
        if goal_idx is not None:
            num_agents = goal_idx.shape[0]

        self.num_agents = num_agents
        self.obstacle_size = obstacle_size
        self.agent_size = agent_size

        map_array = map_str2array(map_str, remove_border, add_border)

        if agent_idx is not None:
            if remove_border:
                agent_idx -= 1

            if add_border:
                agent_idx += 1

            agent_cells = map_array[agent_idx[:, 0], agent_idx[:, 1]]
            assert ~agent_cells.any(), f"agent_idx must be free. got {agent_cells}"

        if goal_idx is not None:
            if remove_border:
                goal_idx -= 1

            if add_border:
                goal_idx += 1

            goal_cells = map_array[goal_idx[:, 0], goal_idx[:, 1]]
            assert ~goal_cells.any(), f"goal_idx must be free. got {goal_cells}"

        self.landmark_pos, free_pos, self.height, self.width = parse_map_array(map_array, obstacle_size)

        if agent_idx is not None:
            agent_pos = idx2pos(agent_idx[:, 0], agent_idx[:, 1], obstacle_size, self.height, self.width)
            self.generate_agents = lambda key: agent_pos
        elif random_agents:
            self.generate_agents = lambda key: jax.random.choice(key, free_pos, shape=(self.num_agents, ), replace=False)
        else:
            agent_pos = jax.random.choice(jax.random.key(0), free_pos, shape=(self.num_agents, ), replace=False)
            self.generate_agents = lambda key: agent_pos

        if goal_idx is not None:
            goal_pos = idx2pos(goal_idx[:, 0], goal_idx[:, 1], obstacle_size, self.height, self.width)
            self.generate_goals = lambda key: goal_pos
        elif random_goals:
            self.generate_goals = lambda key: jax.random.choice(key, free_pos, shape=(self.num_agents, ), replace=False)
            self.generate_goals_lifelong = jax.vmap(lambda key: jax.random.choice(key, free_pos), in_axes=[0]) # 1 key = 1 goal
        else:
            goal_pos = jax.random.choice(jax.random.key(1), goal_pos, shape=(self.num_agents, ), replace=False)
            self.generate_goals = lambda key: goal_pos

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

        key_a, key_g = jax.random.split(key, 2)

        # generate agents
        agent_pos = self.generate_agents(key_a)

        # generate goals
        goal_pos = self.generate_goals(key_g)

        return key_g, self.landmark_pos, agent_pos, goal_pos # return key_g because of lifelong

    @partial(jax.jit, static_argnums=[0])
    def reset_lifelong(self, key) -> Tuple[Array, Array, Array, Array]:
        key_a, key_g = jax.random.split(key, 2)

        # generate agents
        agent_pos = self.generate_agents(key_a)

        # generate goals
        # key for each goal
        key_g = jax.random.split(key_g, self.num_agents)

        goal_pos = self.generate_goals_lifelong(key_g)

        return key_g, self.landmark_pos, agent_pos, goal_pos

    def update_goals(self, keys: ArrayLike, goal_pos: ArrayLike, to_update: ArrayLike) -> Tuple[Array, Array]:
        new_keys = jax.vmap(jax.random.split, in_axes=[0, None])(keys, 1)[:, 0]
        new_keys = jnp.where(to_update, new_keys, keys)

        new_goal_pos = self.generate_goals_lifelong(new_keys)

        return new_keys, new_goal_pos

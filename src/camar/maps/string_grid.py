from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from camar.maps.base import BaseMap
from camar.maps.utils import idx2pos, map_str2array, parse_map_array


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

from typing import Optional, Callable

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from .base_map import base_map
from .const import ENV_DEVICE
from .utils import idx2pos, map_str2array, parse_map_array, random_truncate


class string_grid(base_map):
    def __init__(
        self,
        map_str: str,
        free_pos_str: Optional[str] = None,
        agent_idx: Optional[ArrayLike] = None,
        goal_idx: Optional[ArrayLike] = None,
        num_agents: int = 10,
        random_agents: bool = True,
        random_goals: bool = True,
        remove_border: bool = False,
        add_border: bool = True,
        obstacle_size: float = 0.1,
        agent_size: float = 0.04,
        max_free_pos: Optional[int] = None,
        map_array_preprocess: Callable[[ArrayLike], Array] = lambda map_array: map_array,
        free_pos_array_preprocess: Callable[[ArrayLike], Array] = lambda free_pos_array: free_pos_array,
    ) -> base_map:
        if agent_idx is not None:
            num_agents = agent_idx.shape[0]
        if goal_idx is not None:
            num_agents = goal_idx.shape[0]

        self.num_agents = num_agents
        self.obstacle_size = obstacle_size
        self.agent_size = agent_size

        map_array = map_str2array(
            map_str, remove_border, add_border, map_array_preprocess
        )

        if free_pos_str is not None:
            free_pos_array = map_str2array(
                free_pos_str, remove_border, add_border, free_pos_array_preprocess
            )
        else:
            free_pos_array = None

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

        self.landmark_pos, free_pos, self._height, self._width = parse_map_array(
            map_array, obstacle_size, free_pos_array
        )
        self.landmark_pos = self.landmark_pos.to_device(ENV_DEVICE)

        if max_free_pos is not None:
            free_pos = random_truncate(free_pos, max_free_pos)
        free_pos = free_pos.to_device(ENV_DEVICE)

        if agent_idx is not None:
            agent_pos = idx2pos(
                agent_idx[:, 0], agent_idx[:, 1], obstacle_size, self.height, self.width
            )
            self.generate_agents = lambda key: agent_pos
        elif random_agents:
            self.generate_agents = lambda key: jax.random.choice(
                key, free_pos, shape=(self.num_agents,), replace=False
            )
        else:
            agent_pos = jax.random.choice(
                jax.random.key(0), free_pos, shape=(self.num_agents,), replace=False
            )
            self.generate_agents = lambda key: agent_pos

        if goal_idx is not None:
            goal_pos = idx2pos(
                goal_idx[:, 0], goal_idx[:, 1], obstacle_size, self.height, self.width
            )
            self.generate_goals = lambda key: goal_pos
        elif random_goals:
            self.generate_goals = lambda key: jax.random.choice(
                key, free_pos, shape=(self.num_agents,), replace=False
            )
            self.generate_goals_lifelong = jax.vmap(
                lambda key: jax.random.choice(key, free_pos), in_axes=[0]
            )  # 1 key = 1 goal
        else:
            goal_pos = jax.random.choice(
                jax.random.key(1), free_pos, shape=(self.num_agents,), replace=False
            )
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

    @property
    def height(self):
        return self._height

    @property
    def width(self):
        return self._width

    def reset(self, key: ArrayLike) -> tuple[Array, Array, Array, Array]:
        key_a, key_g = jax.random.split(key, 2)

        # generate agents
        agent_pos = self.generate_agents(key_a)

        # generate goals
        goal_pos = self.generate_goals(key_g)

        return (
            key_g,
            self.landmark_pos,
            agent_pos,
            goal_pos,
        )  # return key_g because of lifelong

    def reset_lifelong(self, key) -> tuple[Array, Array, Array, Array]:
        key_a, key_g = jax.random.split(key, 2)

        # generate agents
        agent_pos = self.generate_agents(key_a)

        # generate goals
        # key for each goal
        key_g = jax.random.split(key_g, self.num_agents)

        goal_pos = self.generate_goals_lifelong(key_g)

        return key_g, self.landmark_pos, agent_pos, goal_pos

    def update_goals(
        self, keys: ArrayLike, goal_pos: ArrayLike, to_update: ArrayLike
    ) -> tuple[Array, Array]:
        new_keys = jax.vmap(jax.random.split, in_axes=[0, None])(keys, 1)[:, 0]
        new_keys = jnp.where(to_update, new_keys, keys)

        new_goal_pos = self.generate_goals_lifelong(new_keys)

        return new_keys, new_goal_pos

from typing import Optional, Callable, Tuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from .base import base_map
from .const import ENV_DEVICE
from .utils import idx2pos, map_str2array, parse_map_array, random_truncate
from camar.registry import register_map


@register_map()
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
        landmark_rad: float = 0.05,
        agent_rad_range: Optional[Tuple[float, float]] = (0.03, 0.03),
        goal_rad_range: Optional[Tuple[float, float]] = None,
        max_free_pos: Optional[int] = None,
        map_array_preprocess: Callable[[ArrayLike], Array] = lambda map_array: map_array,
        free_pos_array_preprocess: Callable[[ArrayLike], Array] = lambda free_pos_array: free_pos_array,
    ) -> base_map:
        if agent_idx is not None:
            num_agents = agent_idx.shape[0]
        if goal_idx is not None:
            num_agents = goal_idx.shape[0]

        self._num_agents = num_agents
        self._landmark_rad = landmark_rad
        self.agent_rad_range = agent_rad_range
        self.goal_rad_range = goal_rad_range

        self.setup_rad()

        map_array = map_str2array(map_str, remove_border, add_border, map_array_preprocess)

        self.map_array = map_array

        if free_pos_str is not None:
            free_pos_array = map_str2array(free_pos_str, remove_border, add_border, free_pos_array_preprocess)
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
            map_array, 2 * self._landmark_rad, free_pos_array
        )
        self.landmark_pos = self.landmark_pos.to_device(ENV_DEVICE)

        if max_free_pos is not None:
            free_pos = random_truncate(free_pos, max_free_pos)
        free_pos = free_pos.to_device(ENV_DEVICE)

        # generate agents
        if agent_idx is not None:
            agent_pos = idx2pos(
                agent_idx[:, 0],
                agent_idx[:, 1],
                2 * self._landmark_rad,
                self.height,
                self.width,
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

        # generate goals
        if goal_idx is not None:
            goal_pos = idx2pos(
                goal_idx[:, 0],
                goal_idx[:, 1],
                2 * self._landmark_rad,
                self.height,
                self.width,
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
            goal_pos = jax.random.choice(jax.random.key(1), free_pos, shape=(self.num_agents,), replace=False)
            self.generate_goals = lambda key: goal_pos

        self._num_landmarks = self.landmark_pos.shape[0]

    def setup_rad(self):
        # Initialize proportional_goal_rad without calling parent's setup_rad
        self.proportional_goal_rad = False
        self.agent_rad = None
        self.landmark_rad = self._landmark_rad
        self.goal_rad = None

        if self.agent_rad_range is not None:
            if self.agent_rad_range[0] == self.agent_rad_range[1]:
                self.agent_rad = self.agent_rad_range[0]
        else:
            self.agent_rad = 0.4 * self._landmark_rad

        if self.goal_rad_range is not None:
            if self.goal_rad_range[0] == self.goal_rad_range[1]:
                self.goal_rad = self.goal_rad_range[0]
            # If min != max, keep self.goal_rad as None to indicate heterogeneous goals
        elif self.agent_rad_range is not None:
            if self.agent_rad_range[0] == self.agent_rad_range[1]:
                self.goal_rad = self.agent_rad / 2.5
            else:
                self.proportional_goal_rad = True
        else:
            self.goal_rad = 0.4 * self._landmark_rad / 2.5

    @property
    def homogeneous_agents(self) -> bool:
        return self.agent_rad is not None

    @property
    def homogeneous_landmarks(self) -> bool:
        return True

    @property
    def homogeneous_goals(self) -> bool:
        return self.goal_rad is not None

    @property
    def num_agents(self) -> int:
        return self._num_agents

    @property
    def num_landmarks(self) -> int:
        return self._num_landmarks

    @property
    def height(self):
        return self._height

    @property
    def width(self):
        return self._width

    def reset(self, key: ArrayLike) -> tuple[Array, Array, Array, Array]:
        key, key_a, key_g = jax.random.split(key, 3)

        # generate agents
        agent_pos = self.generate_agents(key_a)

        # generate goals
        goal_pos = self.generate_goals(key_g)

        sizes = self.generate_sizes(key)

        return (
            key_g,
            self.landmark_pos,
            agent_pos,
            goal_pos,
            sizes,
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

    def update_goals(self, keys: ArrayLike, goal_pos: ArrayLike, to_update: ArrayLike) -> tuple[Array, Array]:
        new_keys = jax.vmap(jax.random.split, in_axes=[0, None])(keys, 1)[:, 0]
        new_keys = jnp.where(to_update, new_keys, keys)

        new_goal_pos = self.generate_goals_lifelong(new_keys)

        return new_keys, new_goal_pos

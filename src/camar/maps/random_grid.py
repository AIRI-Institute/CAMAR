from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from .base import base_map
from .utils import get_border_landmarks
from camar.registry import register_map


@register_map()
class random_grid(base_map):
    def __init__(
        self,
        num_rows: int = 20,
        num_cols: int = 20,
        obstacle_density: float = 0.2,
        num_agents: int = 32,
        grain_factor: int = 3,
        obstacle_size: float = 0.4,
        agent_rad_range: Optional[Tuple[float, float]] = None,
        goal_rad_range: Optional[Tuple[float, float]] = None,
    ) -> base_map:
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.obstacle_density = obstacle_density
        self._num_agents = num_agents
        self.grain_factor = grain_factor
        self.obstacle_size = obstacle_size
        self.agent_rad_range = agent_rad_range
        self.goal_rad_range = goal_rad_range

        self.setup_rad()

        # helpful params
        self.num_obstacles = int(obstacle_density * num_rows * num_cols)
        self._num_landmarks = self.num_obstacles * 4 * (self.grain_factor - 1)
        self._num_landmarks += (num_rows + num_cols) * 2 * (self.grain_factor - 1)  # adding borders

        half_width = self.width / 2
        half_height = self.height / 2

        x_coords = jnp.linspace(
            -half_width + self.obstacle_size / 2,  # start
            half_width - self.obstacle_size / 2,  # end
            num_rows,  # map num_rows
        )
        y_coords = jnp.linspace(
            -half_height + self.obstacle_size / 2,  # start
            half_height - self.obstacle_size / 2,  # end
            num_cols,  # map num_cols
        )

        # coordinates for sampling
        self.map_coordinates = jnp.stack(jnp.meshgrid(x_coords, y_coords), axis=-1).reshape(
            -1, 2
        )  # cell centers of the whole map
        self.border_landmarks = get_border_landmarks(
            num_rows, num_cols, half_width, half_height, self.grain_factor
        )

    def setup_rad(self):
        # default is None
        self.agent_rad = None
        self.goal_rad = None
        self.landmark_rad = None
        self.proportional_goal_rad = False

        # setup landmark_rad
        self.landmark_rad = self.obstacle_size / (2 * (self.grain_factor - 1))

        # setup agent_rad
        if self.agent_rad_range is not None:
            if self.agent_rad_range[0] == self.agent_rad_range[1]:
                self.agent_rad = self.agent_rad_range[0]
        else:
            self.agent_rad = (self.obstacle_size - 2 * self.landmark_rad) * 0.25

        # setup goal_rad
        if self.goal_rad_range is not None:
            if self.goal_rad_range[0] == self.goal_rad_range[1]:
                self.goal_rad = self.goal_rad_range[0]
        # if goal_rad_range is None but agent_rad_range is not None, then goal_rad is proportional to agent_rad
        elif self.agent_rad_range is not None:
            if self.agent_rad_range[0] == self.agent_rad_range[1]:
                self.goal_rad = self.agent_rad / 2.5
            else:
                self.proportional_goal_rad = True
        else:
            self.goal_rad = (self.obstacle_size - 2 * self.landmark_rad) * 0.1

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
        return self.num_cols * self.obstacle_size

    @property
    def width(self):
        return self.num_rows * self.obstacle_size

    def reset(self, key: ArrayLike) -> Tuple[Array, Array, Array, Array]:
        permuted_pos = jax.random.permutation(key, self.map_coordinates)

        agent_pos = jax.lax.dynamic_slice(
            permuted_pos,  # [0 : num_agents, 0 : 2]
            start_indices=(0, 0),
            slice_sizes=(self.num_agents, 2),
        )

        obstacle_pos = jax.lax.dynamic_slice(
            permuted_pos,  # [num_agents : num_agents + num_obstacles, 0 : 2]
            start_indices=(self.num_agents, 0),
            slice_sizes=(self.num_obstacles, 2),
        )

        goal_pos = jax.lax.dynamic_slice(
            permuted_pos,  # [num_agents + num_obstacles : 2 * num_agents + num_obstacles, 0 : 2]
            start_indices=(self.num_agents + self.num_obstacles, 0),
            slice_sizes=(self.num_agents, 2),
        )

        landmark_pos = self.get_landmarks(obstacle_pos, self.grain_factor, self.obstacle_size).reshape(-1, 2)

        all_landmark_pos = jnp.concatenate(
            [
                landmark_pos,
                self.border_landmarks,
            ],
        )

        sizes = self.generate_sizes(key)

        return key, all_landmark_pos, agent_pos, goal_pos, sizes

    @partial(jax.vmap, in_axes=[None, 0, None, None], out_axes=1)
    def get_landmarks(self, obstacle, grain_factor, obstacle_size):
        left_x, down_y = obstacle - obstacle_size / 2
        right_x, up_y = obstacle + obstacle_size / 2

        up_landmarks = jnp.stack(
            (
                jnp.linspace(left_x, right_x, grain_factor - 1, endpoint=False),
                jnp.full((grain_factor - 1,), up_y),
            ),
            axis=-1,
        )
        right_landmarks = jnp.stack(
            (
                jnp.full((grain_factor - 1,), right_x),
                jnp.linspace(up_y, down_y, grain_factor - 1, endpoint=False),
            ),
            axis=-1,
        )
        down_landmarks = jnp.stack(
            (
                jnp.linspace(right_x, left_x, grain_factor - 1, endpoint=False),
                jnp.full((grain_factor - 1,), down_y),
            ),
            axis=-1,
        )
        left_landmarks = jnp.stack(
            (
                jnp.full((grain_factor - 1,), left_x),
                jnp.linspace(down_y, up_y, grain_factor - 1, endpoint=False),
            ),
            axis=-1,
        )

        return jnp.concatenate([up_landmarks, right_landmarks, down_landmarks, left_landmarks])

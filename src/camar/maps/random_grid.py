from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from .base_map import base_map


class random_grid(base_map):
    """The `RandomGrid` class generates a random grid map with obstacles, agents, and goals."""

    def __init__(
        self,
        num_rows: int = 20,
        num_cols: int = 20,
        obstacle_density: float = 0.2,
        num_agents: int = 32,
        grain_factor: int = 6,
        obstacle_size: float = 0.4,
    ) -> base_map:
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.obstacle_density = obstacle_density
        self.num_agents = num_agents
        self.grain_factor = grain_factor
        self.obstacle_size = obstacle_size

        self.width = num_rows * obstacle_size
        self.height = num_cols * obstacle_size

        # helpful params
        self.num_obstacles = int(obstacle_density * num_rows * num_cols)
        self.num_landmarks = self.num_obstacles * 4 * (self.grain_factor - 1)
        self.num_landmarks += (num_rows + num_cols) * 2 * (self.grain_factor - 1) # adding borders

        half_width = self.width / 2
        half_height = self.height / 2

        x_coords = jnp.linspace(
            - half_width + self.obstacle_size / 2, # start
            half_width - self.obstacle_size / 2, # end
            num_rows, # map num_rows
        )
        y_coords = jnp.linspace(
            - half_height + self.obstacle_size / 2, # start
            half_height - self.obstacle_size / 2, # end
            num_cols, # map num_cols
        )

        # coordinates for sampling
        self.map_coordinates = jnp.stack(jnp.meshgrid(x_coords, y_coords), axis=-1).reshape(-1, 2) # cell centers of the whole map
        self.border_landmarks = self.get_border_landmarks(num_rows, num_cols, half_width, half_height, self.grain_factor)

    @property
    def landmark_rad(self) -> float:
        return self.obstacle_size / (2 * (self.grain_factor - 1))

    @property
    def agent_rad(self):
        return (self.obstacle_size - 2 * self.landmark_rad) * 0.4

    @property
    def goal_rad(self):
        return self.agent_rad / 2.5

    @partial(jax.jit, static_argnums=[0])
    def reset(self, key: ArrayLike) -> Tuple[Array, Array, Array, Array]:
        permuted_pos = jax.random.permutation(key, self.map_coordinates)

        agent_pos = jax.lax.dynamic_slice(permuted_pos, # [0 : num_agents, 0 : 2]
                                          start_indices=(0,               0),
                                          slice_sizes=  (self.num_agents, 2))

        obstacle_pos = jax.lax.dynamic_slice(permuted_pos, # [num_agents : num_agents + num_obstacles, 0 : 2]
                                             start_indices=(self.num_agents,    0),
                                             slice_sizes=  (self.num_obstacles, 2))

        goal_pos = jax.lax.dynamic_slice(permuted_pos, # [num_agents + num_obstacles : 2 * num_agents + num_obstacles, 0 : 2]
                                         start_indices=(self.num_agents + self.num_obstacles, 0),
                                         slice_sizes=  (self.num_agents,                      2))

        landmark_pos = self.get_landmarks(obstacle_pos, self.grain_factor, self.obstacle_size).reshape(-1, 2)

        all_landmark_pos = jnp.concatenate(
            [
                landmark_pos,
                self.border_landmarks,
            ],
        )

        return key, all_landmark_pos, agent_pos, goal_pos

    def get_border_landmarks(self, num_rows, num_cols, half_width, half_height, grain_factor):
        top_wall = jnp.stack(
            (
                jnp.linspace(- half_width, # start
                             half_width, # end
                             num_rows * (grain_factor - 1), # num points
                             endpoint=False),
                jnp.full((num_rows * (grain_factor - 1), ), # num points
                         half_height), # y coord of the top wall
            ),
            axis=-1,
        )
        right_wall = jnp.stack(
            (
                jnp.full((num_cols * (grain_factor - 1), ), # num points
                         half_width), # x coord of the right wall
                jnp.linspace(half_height, # start
                             - half_height, # end
                             num_cols * (grain_factor - 1), # num points
                             endpoint=False),
            ),
            axis=-1,
        )
        bottom_wall = jnp.stack(
            (
                jnp.linspace(half_width, # start
                             - half_width, # end
                             num_rows * (grain_factor - 1), # num points
                             endpoint=False),
                jnp.full((num_rows * (grain_factor - 1), ), # num points
                         - half_height), # y coord of the bottom wall
            ),
            axis=-1,
        )
        left_wall = jnp.stack(
            (
                jnp.full((num_cols * (grain_factor - 1), ), # num points
                         - half_width), # x coord of the left wall
                jnp.linspace(- half_height, # start
                             half_height, # end
                             num_cols * (grain_factor - 1), # num points
                             endpoint=False),
            ),
            axis=-1,
        )
        return jnp.concatenate([top_wall,
                                right_wall,
                                left_wall,
                                bottom_wall])

    @partial(jax.vmap, in_axes=[None, 0, None, None], out_axes=1)
    def get_landmarks(self, obstacle, grain_factor, obstacle_size):
        left_x, down_y = obstacle - obstacle_size / 2
        right_x, up_y = obstacle + obstacle_size / 2

        up_landmarks = jnp.stack(
            (jnp.linspace(left_x, right_x, grain_factor - 1, endpoint=False),
             jnp.full((grain_factor - 1, ), up_y)),
            axis=-1)
        right_landmarks = jnp.stack(
            (jnp.full((grain_factor - 1, ), right_x),
             jnp.linspace(up_y, down_y, grain_factor - 1, endpoint=False)),
            axis=-1)
        down_landmarks = jnp.stack(
            (jnp.linspace(right_x, left_x, grain_factor - 1, endpoint=False),
             jnp.full((grain_factor - 1, ), down_y)),
            axis=-1)
        left_landmarks = jnp.stack(
            (jnp.full((grain_factor - 1, ), left_x),
             jnp.linspace(down_y, up_y, grain_factor - 1, endpoint=False)),
            axis=-1)

        return jnp.concatenate([up_landmarks,
                                right_landmarks,
                                down_landmarks,
                                left_landmarks])

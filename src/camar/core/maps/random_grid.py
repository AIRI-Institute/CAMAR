from functools import partial

import jax
import jax.numpy as jnp
from base import Map
from jax import Array
from jax.typing import ArrayLike


class RandomGrid(Map):
    """The `RandomGrid` class generates a random grid map with obstacles, agents, and goals."""

    def __init__(
        self,
        width: int = 19,
        height: int = 19,
        obstacle_density: float = 0.3,
        num_agents: int = 8,
        grain_factor: int = 4,
        obstacle_size: float = 4,
    ) -> Map:
        self.width = width
        self.height = height
        self.obstacle_density = obstacle_density
        self.num_agents = num_agents
        self.grain_factor = grain_factor
        self.obstacle_size = obstacle_size
        self.add_border = True

        # helpful params
        self.num_obstacles = int(obstacle_density * width * height)
        self.num_landmarks = self.num_obstacles * 4 * (self.grain_factor - 1)
        self.num_landmarks += (width + height) * 2 * (self.grain_factor - 1) # adding borders

        half_width = width * self.obstacle_size / 2
        half_height = height * self.obstacle_size / 2

        x_coords = jnp.linspace(
            - half_width + self.obstacle_size / 2, # start
            half_width - self.obstacle_size / 2, # end
            width, # map width
        )
        y_coords = jnp.linspace(
            - half_height + self.obstacle_size / 2, # start
            half_height - self.obstacle_size / 2, # end
            height, # map height
        )

        # coordinates for sampling
        self.map_coordinates = jnp.stack(jnp.meshgrid(x_coords, y_coords), axis=-1).reshape(-1, 2) # cell centers of the whole map
        self.border_landmarks = self.get_border_landmarks(width, height, half_width, half_height, self.grain_factor)
    
    @property
    def landmark_rad(self) -> float:  # noqa: D102
        return self.obstacle_size / (2 * (self.grain_factor - 1))

    @property
    def agent_rad(self):
        return (self.obstacle_size - 2 * self.landmark_rad) * 0.4
    
    @property
    def goal_rad(self):
        return self.obstacle_size / 10
    
    def reset(self, key: ArrayLike) -> Array:
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

        return all_landmark_pos, agent_pos, goal_pos
    
    def get_border_landmarks(self, width, height, half_width, half_height, grain_factor):
        top_wall = jnp.stack(
            (
                jnp.linspace(- half_width, # start
                             half_width, # end
                             width * (grain_factor - 1), # num points
                             endpoint=False),
                jnp.full((width * (grain_factor - 1), ), # num points
                         half_height), # y coord of the top wall
            ),
            axis=-1,
        )
        right_wall = jnp.stack(
            (
                jnp.full((height * (grain_factor - 1), ), # num points
                         half_width), # x coord of the right wall
                jnp.linspace(half_height, # start
                             - half_height, # end
                             height * (grain_factor - 1), # num points
                             endpoint=False),
            ),
            axis=-1,
        )
        bottom_wall = jnp.stack(
            (
                jnp.linspace(half_width, # start
                             - half_width, # end
                             width * (grain_factor - 1), # num points
                             endpoint=False),
                jnp.full((width * (grain_factor - 1), ), # num points
                         - half_height), # y coord of the bottom wall
            ),
            axis=-1,
        )
        left_wall = jnp.stack(
            (
                jnp.full((height * (grain_factor - 1), ), # num points
                         - half_width), # x coord of the left wall
                jnp.linspace(- half_height, # start
                             half_height, # end
                             height * (grain_factor - 1), # num points
                             endpoint=False),
            ),
            axis=-1,
        )
        return jnp.concatenate([top_wall,
                                right_wall,
                                left_wall,
                                bottom_wall])
    
    @partial(jax.vmap, in_axes=[0, None, None], out_axes=1)
    def get_landmarks(obstacle, grain_factor, obstacle_size):
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

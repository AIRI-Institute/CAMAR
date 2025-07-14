import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from .base import base_map
from .utils import get_border_landmarks, idx2pos, perlin_noise_vectorized


class caves_cont(base_map):
    def __init__(
        self,
        num_rows: int = 128,
        num_cols: int = 128,
        scale: int = 14,
        landmark_low_ratio: float = 0.55,
        landmark_high_ratio: float = 0.72,
        free_ratio: float = 0.20,
        add_borders: bool = True,
        num_agents: int = 16,
        obstacle_size: float = 0.1,
        agent_size: float = 0.2,
    ):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.obstacle_size = obstacle_size
        self.agent_size = agent_size

        self.grid_num_rows = int(num_rows / scale)
        self.grid_num_cols = int(num_cols / scale)

        if landmark_low_ratio >= landmark_high_ratio:
            raise ValueError("0th element of landmark_ranks must be less than 1th.")

        num_cells = num_rows * num_cols
        self.landmark_ranks = (
            int(num_cells * landmark_low_ratio),
            int(num_cells * landmark_high_ratio),
        )
        self.num_landmarks = self.landmark_ranks[1] - self.landmark_ranks[0]

        self.free_rank = int(num_cells * free_ratio)

        self.num_agents = num_agents

        if add_borders:
            grain_factor = 2
            self.border_landmarks = get_border_landmarks(
                num_rows,
                num_cols,
                half_width=self.width / 2,
                half_height=self.height / 2,
                grain_factor=grain_factor,
            )
            self.num_landmarks += (num_rows + num_cols) * 2 * (grain_factor - 1)
        else:
            self.border_landmarks = jnp.empty(shape=(0, 2))

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
        return self.num_cols * self.obstacle_size

    @property
    def width(self):
        return self.num_rows * self.obstacle_size

    def reset(self, key: ArrayLike) -> tuple[Array, Array, Array, Array]:
        key_o, key_a, key_g = jax.random.split(key, 3)

        # generate perlin noise
        noise = perlin_noise_vectorized(
            key_o, self.num_cols, self.num_rows, self.grid_num_cols, self.grid_num_rows
        )

        noise = jnp.abs(noise).flatten()

        # extract landmarks
        landmark_idx_high = jnp.argpartition(noise, self.landmark_ranks[1])[
            : self.landmark_ranks[1]
        ]
        landmark_idx_low = jnp.argpartition(noise, self.landmark_ranks[0])[
            : self.landmark_ranks[0]
        ]
        landmark_idx = jnp.setdiff1d(
            landmark_idx_high,
            landmark_idx_low,
            size=self.landmark_ranks[1] - self.landmark_ranks[0],
        )
        landmark_idx_x, landmark_idx_y = jnp.divmod(landmark_idx, self.num_cols)

        landmark_pos = idx2pos(
            landmark_idx_x, landmark_idx_y, self.obstacle_size, self.height, self.width
        )

        # add borders
        landmark_pos = jnp.vstack((landmark_pos, self.border_landmarks))

        # extract free pos
        free_idx = jnp.argpartition(noise, self.free_rank)[: self.free_rank]
        free_idx_x, free_idx_y = jnp.divmod(free_idx, self.num_cols)

        free_pos = idx2pos(
            free_idx_x, free_idx_y, self.obstacle_size, self.height, self.width
        )

        # generate agents
        agent_pos = jax.random.choice(
            key_a, free_pos, shape=(self.num_agents,), replace=False
        )

        # generate goals
        goal_pos = jax.random.choice(
            key_g, free_pos, shape=(self.num_agents,), replace=False
        )

        return (
            key_g,
            landmark_pos,
            agent_pos,
            goal_pos,
        )  # return key_g because of lifelong

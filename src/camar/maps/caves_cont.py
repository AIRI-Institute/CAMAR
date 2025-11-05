import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from typing import Optional, Tuple

from .base import base_map
from .utils import get_border_landmarks, idx2pos, perlin_noise_vectorized
from camar.registry import register_map


@register_map()
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
        landmark_rad: float = 0.05,
        agent_rad_range: Optional[Tuple[float, float]] = (0.1, 0.1),
        goal_rad_range: Optional[Tuple[float, float]] = None,
    ):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self._landmark_rad = landmark_rad
        self.agent_rad_range = agent_rad_range
        self.goal_rad_range = goal_rad_range

        self.setup_rad()

        self.grid_num_rows = int(num_rows / scale)
        self.grid_num_cols = int(num_cols / scale)

        if landmark_low_ratio >= landmark_high_ratio:
            raise ValueError("0th element of landmark_ranks must be less than 1th.")

        num_cells = num_rows * num_cols
        self.landmark_ranks = (
            int(num_cells * landmark_low_ratio),
            int(num_cells * landmark_high_ratio),
        )
        self._num_landmarks = self.landmark_ranks[1] - self.landmark_ranks[0]

        self.free_rank = int(num_cells * free_ratio)

        self._num_agents = num_agents

        if add_borders:
            grain_factor = 2
            self.border_landmarks = get_border_landmarks(
                num_rows,
                num_cols,
                half_width=self.width / 2,
                half_height=self.height / 2,
                grain_factor=grain_factor,
            )
            self._num_landmarks += (num_rows + num_cols) * 2 * (grain_factor - 1)
        else:
            self.border_landmarks = jnp.empty(shape=(0, 2))

    def setup_rad(self):
        # default is None
        self.agent_rad = None
        self.goal_rad = None
        self.landmark_rad = None
        self.proportional_goal_rad = False

        # setup landmark_rad
        self.landmark_rad = self._landmark_rad

        # setup agent_rad
        if self.agent_rad_range is not None:
            if self.agent_rad_range[0] == self.agent_rad_range[1]:
                self.agent_rad = self.agent_rad_range[0]
        else:
            self.agent_rad = 0.4 * self._landmark_rad

        # setup goal_rad
        if self.goal_rad_range is not None:
            if self.goal_rad_range[0] == self.goal_rad_range[1]:
                self.goal_rad = self.goal_rad_range[0]
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
        return self.landmark_rad is not None

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
        return self.num_cols * self.landmark_rad * 2

    @property
    def width(self):
        return self.num_rows * self.landmark_rad * 2

    def reset(self, key: ArrayLike) -> tuple[Array, Array, Array, Array]:
        key, key_o, key_a, key_g = jax.random.split(key, 4)

        # generate perlin noise
        noise = perlin_noise_vectorized(
            key_o, self.num_cols, self.num_rows, self.grid_num_cols, self.grid_num_rows
        )

        noise = jnp.abs(noise).flatten()

        # extract landmarks
        landmark_idx_high = jnp.argpartition(noise, self.landmark_ranks[1])[: self.landmark_ranks[1]]
        landmark_idx_low = jnp.argpartition(noise, self.landmark_ranks[0])[: self.landmark_ranks[0]]
        landmark_idx = jnp.setdiff1d(
            landmark_idx_high,
            landmark_idx_low,
            size=self.landmark_ranks[1] - self.landmark_ranks[0],
        )
        landmark_idx_x, landmark_idx_y = jnp.divmod(landmark_idx, self.num_cols)

        landmark_pos = idx2pos(
            landmark_idx_x,
            landmark_idx_y,
            2 * self.landmark_rad,
            self.height,
            self.width,
        )

        # add borders
        landmark_pos = jnp.vstack((landmark_pos, self.border_landmarks))

        # extract free pos
        free_idx = jnp.argpartition(noise, self.free_rank)[: self.free_rank]
        free_idx_x, free_idx_y = jnp.divmod(free_idx, self.num_cols)

        free_pos = idx2pos(free_idx_x, free_idx_y, 2 * self.landmark_rad, self.height, self.width)

        # generate agents
        agent_pos = jax.random.choice(key_a, free_pos, shape=(self.num_agents,), replace=False)

        # generate goals
        goal_pos = jax.random.choice(key_g, free_pos, shape=(self.num_agents,), replace=False)

        sizes = self.generate_sizes(key)

        return (
            key_g,
            landmark_pos,
            agent_pos,
            goal_pos,
            sizes,
        )  # return key_g because of lifelong

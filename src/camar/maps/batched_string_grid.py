from typing import List, Optional, Callable, Tuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from .base import base_map
from .const import ENV_DEVICE
from .utils import (
    check_pos,
    idx2pos,
    map_str2array,
    pad_placeholder,
    parse_map_array,
    random_truncate,
)
from camar.registry import register_map


@register_map()
class batched_string_grid(base_map):
    def __init__(
        self,
        map_str_batch: List[str],
        free_pos_str_batch: Optional[List[str]] = None,
        agent_idx_batch: Optional[List[ArrayLike]] = None,
        goal_idx_batch: Optional[List[ArrayLike]] = None,
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
        self.batch_size = len(map_str_batch)
        if agent_idx_batch is not None:
            num_agents = agent_idx_batch[0].shape[0]

            # check agent_idx_batch
            assert all(map(lambda agent_idx: agent_idx.shape[0] == num_agents, agent_idx_batch)), (
                "agent_idx.shape must be the same in a batch."
            )
            assert len(agent_idx_batch) == self.batch_size

        if goal_idx_batch is not None:
            num_agents = goal_idx_batch[0].shape[0]

            # check goal_idx_batch
            assert all(map(lambda goal_idx: goal_idx.shape[0] == num_agents, goal_idx_batch)), (
                "goal_idx.shape must be the same in a batch."
            )
            assert len(goal_idx_batch) == self.batch_size

        self._num_agents = num_agents
        self._landmark_rad = landmark_rad
        self.agent_rad_range = agent_rad_range
        self.goal_rad_range = goal_rad_range

        self.setup_rad()

        map_array_batch = list(
            map(
                lambda map_str: map_str2array(
                    map_str,
                    remove_border,
                    add_border,
                    map_array_preprocess,
                ),
                map_str_batch,
            )
        )

        # check map_array_batch
        map_array_shape = map_array_batch[0].shape
        map_array_checks = map(lambda map_array: map_array.shape == map_array_shape, map_array_batch)
        assert all(map_array_checks), (
            "all maps must have the same shape.Resize or provide resize map_array_preprocess."
        )

        if free_pos_str_batch is not None:
            free_pos_array_batch = list(
                map(
                    lambda free_pos_str: map_str2array(
                        free_pos_str,
                        remove_border,
                        add_border,
                        free_pos_array_preprocess,
                    ),
                    free_pos_str_batch,
                )
            )

            # check free_pos_array_batch
            free_pos_array_checks = map(
                lambda map_array, free_pos_array: map_array.shape == free_pos_array.shape,
                map_array_batch,
                free_pos_array_batch,
            )
            assert all(free_pos_array_checks), (
                "all free_pos must have the same shape as maps after processing."
                "Resize or provide resize free_pos_array_preprocess."
            )
        else:
            free_pos_array_batch = [None for _ in map_str_batch]

        if agent_idx_batch is not None:
            if remove_border:
                agent_idx_batch = [idx - 1 for idx in agent_idx_batch]

            if add_border:
                agent_idx_batch = [idx + 1 for idx in agent_idx_batch]

            # check if agents on free cells
            agent_checks = map(
                lambda map_array, agent_idx: check_pos(map_array, agent_idx),
                map_array_batch,
                agent_idx_batch,
            )
            assert any(agent_checks), "agent_idx must be free for each map instance."

        if goal_idx_batch is not None:
            if remove_border:
                goal_idx_batch = [idx - 1 for idx in goal_idx_batch]

            if add_border:
                goal_idx_batch = [idx + 1 for idx in goal_idx_batch]

            # check if goals on free cells
            goal_checks = map(
                lambda map_array, goal_idx: check_pos(map_array, goal_idx),
                map_array_batch,
                goal_idx_batch,
            )
            assert any(goal_checks), "goal_idx must be free for each map instance."

        self.landmark_pos_batch, free_pos_batch, height_batch, width_batch = zip(
            *map(
                lambda map_array, free_pos_array: parse_map_array(
                    map_array, 2 * self._landmark_rad, free_pos_array
                ),
                map_array_batch,
                free_pos_array_batch,
            )
        )

        # check height in a batch
        self._height = height_batch[0]
        assert all(map(lambda height: height == self._height, height_batch)), (
            "map height must be the same in a batch."
        )

        # check width in a batch
        self._width = width_batch[0]
        assert all(map(lambda width: width == self._width, width_batch)), (
            "map width must be the same in a batch."
        )

        self._num_landmarks = max(map(lambda landmark_pos: landmark_pos.shape[0], self.landmark_pos_batch))
        self.free_pos_num = min(map(lambda free_pos: free_pos.shape[0], free_pos_batch))

        if max_free_pos is not None:
            self.free_pos_num = min(self.free_pos_num, max_free_pos)  # for memory control

        # check free cells is enough for all agents
        assert self.free_pos_num >= self.num_agents, (
            "there is a map without enough number of free cells for agents"
        )

        self.landmark_pos_batch = jnp.stack(
            list(
                map(
                    lambda landmark_pos: pad_placeholder(landmark_pos, self.num_landmarks),
                    self.landmark_pos_batch,
                )
            ),
            axis=0,
        )

        self.landmark_pos_batch = self.landmark_pos_batch.to_device(ENV_DEVICE)

        free_pos_batch = jnp.stack(
            list(
                map(
                    lambda free_pos: random_truncate(free_pos, self.free_pos_num),
                    free_pos_batch,
                )
            ),
            axis=0,
        )

        free_pos_batch = free_pos_batch.to_device(ENV_DEVICE)

        if agent_idx_batch is not None:
            agent_pos_batch = jax.vmap(idx2pos, in_axes=[0, 0, None, None, None])(
                jnp.array(agent_idx_batch)[:, :, 0],
                jnp.array(agent_idx_batch)[:, :, 1],
                2 * self._landmark_rad,
                self.height,
                self.width,
            )
            self.generate_agents = lambda key_batch, key_a: jax.random.choice(key_batch, agent_pos_batch)

        elif random_agents:

            def generate_agents(key_batch, key_a):
                free_pos = jax.random.choice(key_batch, free_pos_batch)
                agent_pos = jax.random.choice(key_a, free_pos, shape=(self.num_agents,), replace=False)
                return agent_pos

            self.generate_agents = generate_agents

        else:
            agent_pos_batch = jax.random.choice(
                jax.random.key(0),
                free_pos_batch,
                shape=(self.num_agents,),
                replace=False,
                axis=1,
            )
            self.generate_agents = lambda key_batch, key_a: jax.random.choice(key_batch, agent_pos_batch)

        if goal_idx_batch is not None:
            goal_pos_batch = jax.vmap(idx2pos, in_axes=[0, 0, None, None, None])(
                jnp.array(goal_idx_batch)[:, :, 0],
                jnp.array(goal_idx_batch)[:, :, 1],
                2 * self._landmark_rad,
                self.height,
                self.width,
            )
            self.generate_goals = lambda key_batch, key_a: jax.random.choice(key_batch, goal_pos_batch)

        elif random_goals:

            def generate_goals(key_batch, key_g):
                free_pos = jax.random.choice(key_batch, free_pos_batch)
                goal_pos = jax.random.choice(key_g, free_pos, shape=(self.num_agents,), replace=False)
                return goal_pos

            self.generate_goals = generate_goals

        else:
            goal_pos_batch = jax.random.choice(
                jax.random.key(0),
                free_pos_batch,
                shape=(self.num_agents,),
                replace=False,
                axis=1,
            )
            self.generate_goals = lambda key_batch, key_g: jax.random.choice(key_batch, goal_pos_batch)

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
        # if goal_rad_range is None but agent_rad_range is not None, then goal_rad is proportional to agent_rad
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
        key_batch, key = jax.random.split(key, 2)

        key, key_a, key_g = jax.random.split(key, 3)

        # generate agents
        agent_pos = self.generate_agents(key_batch, key_a)

        # generate goals
        goal_pos = self.generate_goals(key_batch, key_g)

        landmark_pos = jax.random.choice(key_batch, self.landmark_pos_batch)

        sizes = self.generate_sizes(key)

        return key_g, landmark_pos, agent_pos, goal_pos, sizes

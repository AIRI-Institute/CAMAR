from functools import partial
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from camar.maps.base import BaseMap
from camar.maps.utils import (
    check_pos,
    idx2pos,
    map_str2array,
    pad_placeholder,
    parse_map_array,
    random_truncate,
)


class BatchedStringGrid(BaseMap):
    def __init__(
        self,
        map_str_batch: List[str],
        agent_idx_batch: Optional[List[ArrayLike]] = None,
        goal_idx_batch: Optional[List[ArrayLike]] = None,
        num_agents: Optional[int] = 10,
        random_agents: Optional[bool] = True,
        random_goals: Optional[bool] = True,
        remove_border: bool = False,
        add_border: bool = True,
        obstacle_size: float = 0.1,
        agent_size: float = 0.04,
    ) -> BaseMap:
        self.batch_size = len(map_str_batch)
        if agent_idx_batch is not None:
            num_agents = agent_idx_batch[0].shape[0]
            assert all(map(lambda x: x.shape[0] == num_agents, agent_idx_batch)), "agent_idx.shape must be the same in a batch."
            assert len(agent_idx_batch) == self.batch_size
        if goal_idx_batch is not None:
            num_agents = goal_idx_batch[0].shape[0]
            assert all(map(lambda x: x.shape[0] == num_agents, goal_idx_batch)), "goal_idx.shape must be the same in a batch."
            assert len(goal_idx_batch) == self.batch_size

        self.num_agents = num_agents
        self.obstacle_size = obstacle_size
        self.agent_size = agent_size

        map_array_batch = list(map(lambda x: map_str2array(x, remove_border, add_border), map_str_batch))

        if agent_idx_batch is not None:
            if remove_border:
                agent_idx_batch = [idx - 1 for idx in agent_idx_batch]

            if add_border:
                agent_idx_batch = [idx + 1 for idx in agent_idx_batch]

            agent_checks = map(lambda x, y: check_pos(x, y), map_array_batch, agent_idx_batch)
            assert any(agent_checks), "agent_idx must be free for each map instance."

        if goal_idx_batch is not None:
            if remove_border:
                goal_idx_batch = [idx - 1 for idx in goal_idx_batch]

            if add_border:
                goal_idx_batch = [idx + 1 for idx in goal_idx_batch]

            goal_checks = map(lambda x, y: check_pos(x, y), map_array_batch, goal_idx_batch)
            assert any(goal_checks), "goal_idx must be free for each map instance."

        self.landmark_pos_batch, free_pos_batch, height_batch, width_batch = list(zip(*map(lambda x: parse_map_array(x, obstacle_size), map_array_batch)))
        self.height = height_batch[0]
        assert all(map(lambda x: x == self.height, height_batch)), "map height must be the same in a batch."

        self.width = width_batch[0]
        assert all(map(lambda x: x == self.width, width_batch)), "map width must be the same in a batch."

        self.num_landmarks = max(map(lambda x: x.shape[0], self.landmark_pos_batch))
        self.free_pos_num = min(map(lambda x: x.shape[0], free_pos_batch))

        assert self.free_pos_num >= self.num_agents, "there is a map without enough number of free cells for agents"

        self.landmark_pos_batch = jnp.stack(list(map(lambda x: pad_placeholder(x, self.num_landmarks), self.landmark_pos_batch)), axis=0)

        free_pos_batch = jnp.stack(list(map(lambda x: random_truncate(x, self.free_pos_num), free_pos_batch)), axis=0)

        if agent_idx_batch is not None:
            agent_pos_batch = jax.vmap(idx2pos, in_axes=[0, 0, None, None, None])(agent_idx_batch[:, :, 0], agent_idx_batch[:, :, 1], obstacle_size, self.height, self.width)
            self.generate_agents = lambda key_batch, key_a: jax.random.choice(key_batch, agent_pos_batch)
        elif random_agents:
            @jax.jit
            def generate_agents(key_batch, key_a):
                free_pos = jax.random.choice(key_batch, free_pos_batch)
                agent_pos = jax.random.choice(key_a, free_pos, shape=(self.num_agents, ), replace=False)
                return agent_pos

            self.generate_agents = generate_agents
        else:
            agent_pos_batch = jax.random.choice(jax.random.key(0), free_pos_batch, shape=(self.num_agents, ), replace=False, axis=1)
            self.generate_agents = lambda key_batch, key_a: jax.random.choice(key_batch, agent_pos_batch)

        if goal_idx_batch is not None:
            goal_pos_batch = jax.vmap(idx2pos, in_axes=[0, 0, None, None, None])(goal_idx_batch[:, :, 0], goal_idx_batch[:, :, 1], obstacle_size, self.height, self.width)
            self.generate_goals = lambda key_batch, key_a: jax.random.choice(key_batch, goal_pos_batch)
        elif random_goals:
            @jax.jit
            def generate_goals(key_batch, key_g):
                free_pos = jax.random.choice(key_batch, free_pos_batch)
                goal_pos = jax.random.choice(key_g, free_pos, shape=(self.num_agents, ), replace=False)
                return goal_pos

            self.generate_goals = generate_goals
        else:
            goal_pos_batch = jax.random.choice(jax.random.key(0), free_pos_batch, shape=(self.num_agents, ), replace=False, axis=1)
            self.generate_goals = lambda key_batch, key_g: jax.random.choice(key_batch, goal_pos_batch)

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

        key_batch, key = jax.random.split(key, 2)

        key_a, key_g = jax.random.split(key, 2)

        # generate agents
        agent_pos = self.generate_agents(key_batch, key_a)

        # generate goals
        goal_pos = self.generate_goals(key_batch, key_g)

        landmark_pos = jax.random.choice(key_batch, self.landmark_pos_batch)

        return key_g, landmark_pos, agent_pos, goal_pos

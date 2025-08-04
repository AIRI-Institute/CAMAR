import jax
from flax import struct
from jax import Array
from jax import numpy as jnp
from jax.typing import ArrayLike

from camar.utils import State
from camar.wrappers import GymnaxWrapper
from rrt import RRT, RRTState


@struct.dataclass
class State_with_RRTState:
    env_state: State
    rrt_state: RRTState

    rrt_cur_idx: Array
    rrt_cur_goal_pos: Array

    def __getattr__(self, name):
        if hasattr(self.env_state, name):
            return getattr(self.env_state, name)
        elif hasattr(self.rrt_state, name):
            return getattr(self.rrt_state, name)
        elif hasattr(super(), name):
            return super().__getattr__(self, name)
        else:
            raise AttributeError(name)


class RRTWrapper(GymnaxWrapper):
    def __init__(self, env, rrt_num_iters, rrt_goal_rad=None, rrt_step_size=None):
        super().__init__(env)

        if rrt_step_size is None:
            rrt_step_size = env.agent_rad

        self.rrt_goal_rad = rrt_goal_rad
        if rrt_goal_rad is None:
            self.rrt_goal_rad = env.agent_rad

        self._rrt = RRT(env, num_samples=rrt_num_iters, step_size=rrt_step_size, goal_rad=env.agent_rad)

        self.observation_size = self._env.observation_size + 2

    def reset(self, key: ArrayLike):
        key_env, key_rrt = jax.random.split(key, 2)
        obs, env_state = self._env.reset(key_env)

        rrt_state = self._rrt.run(
            key_rrt, start=env_state.goal_pos, goal=env_state.agent_pos, landmark_pos=env_state.landmark_pos
        )

        last_idx = self._rrt.find_last_idx(rrt_state)

        rrt_idx = jnp.where(rrt_state.goal_reached & (last_idx != -1), last_idx, 0)

        rrt_goal_pos = rrt_state.pos[rrt_idx, jnp.arange(self.num_agents), :]  # (num_agents, 2)

        state = State_with_RRTState(
            env_state=env_state,
            rrt_state=rrt_state,
            rrt_cur_idx=last_idx,
            rrt_cur_goal_pos=rrt_goal_pos,
        )

        rrt_goal_norm = self.get_obs_rrt_goal(rrt_goal_pos, env_state.agent_pos)

        new_obs = jnp.concatenate((rrt_goal_norm, obs), axis=1)

        return new_obs, state

    def step(self, key: ArrayLike, state: State, actions: ArrayLike):
        old_rrt_goal_dist = jnp.linalg.norm(
            state.rrt_cur_goal_pos - state.agent_pos, axis=-1
        )  # (num_agents, )

        obs, env_state, reward, done, info = self._env.step(key, state.env_state, actions)

        rrt_goal_dist = jnp.linalg.norm(env_state.agent_pos - state.rrt_cur_goal_pos, axis=-1)

        rrt_on_goal = rrt_goal_dist <= self.rrt_goal_rad

        rrt_cur_idx = jnp.where(
            rrt_on_goal, state.parent[state.rrt_cur_idx, jnp.arange(self.num_agents)], state.rrt_cur_idx
        )

        rrt_idx = jnp.where(state.goal_reached & (rrt_cur_idx != -1), rrt_cur_idx, 0)

        rrt_goal_pos = state.pos[rrt_idx, jnp.arange(self.num_agents), :]  # (num_agents, 2)

        state = state.replace(
            env_state=env_state,
            rrt_cur_idx=rrt_cur_idx,
            rrt_cur_goal_pos=rrt_goal_pos,
        )

        rrt_goal_norm = self.get_obs_rrt_goal(rrt_goal_pos, env_state.agent_pos)

        new_obs = jnp.concatenate((rrt_goal_norm, obs), axis=1)

        reward = (
            0.5 * env_state.on_goal.astype(jnp.float32)
            + 0.05 * rrt_on_goal.astype(jnp.float32)
            - 1.0 * env_state.is_collision.astype(jnp.float32)
            + self.pos_shaping_factor * (old_rrt_goal_dist - rrt_goal_dist)
        )
        reward = reward.reshape(self.num_agents, 1)

        return new_obs, state, reward, done, info

    def get_obs_rrt_goal(self, goal_pos, agent_pos):
        ego_goal = goal_pos - agent_pos  # (num_agents, 2)

        goal_dist = jnp.linalg.norm(ego_goal, axis=-1)  # (num_agents, )

        ego_goal_norm = jnp.where(goal_dist[:, None] > 1.0, ego_goal / goal_dist[:, None], ego_goal)

        return ego_goal_norm

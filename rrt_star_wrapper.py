import jax
from flax import struct
from jax import numpy as jnp
from jax.typing import ArrayLike

from camar.environment import Camar
from camar.utils import State
from camar.wrappers import GymnaxWrapper
from rrt_star import RRTStar, RRTStarState


@struct.dataclass
class State_with_RRTStarState:
    env_state: State
    rrt_state: RRTStarState

    def __getattr__(self, name):
        if hasattr(self.env_state, name):
            return getattr(self.env_state, name)
        elif hasattr(self.rrt_state, name):
            return getattr(self.rrt_state, name)
        elif hasattr(super(), name):
            return super().__getattr__(self, name)
        else:
            raise AttributeError(name)


class RRTStarWrapper(GymnaxWrapper, Camar):
    def __init__(
        self, env, rrt_num_iters, num_obs_points, rrt_step_size=None, num_neighbours=10, goal_rad=None
    ):
        super().__init__(env)

        self.num_obs_points = num_obs_points

        self.rrt_goal_rad = (
            goal_rad or env.map_generator.agent_rad * 2
        )  # only for goal_reached flag == useless

        self.rrt_step_size = rrt_step_size or env.map_generator.landmark_rad

        self._rrt = RRTStar(
            env=env,
            num_samples=rrt_num_iters,
            step_size=self.rrt_step_size,
            num_neighbours=num_neighbours,
            goal_rad=self.rrt_goal_rad,
        )

    @property
    def observation_size(self):
        return self.num_obs_points * 3 + super().observation_size

    def get_nearest_cost(self, state):
        pos = state.rrt_state.pos  # (num_samples, num_agents, 2)
        cost = state.rrt_state.cost  # (num_samples, num_agents)
        parent = state.rrt_state.parent  # (num_samples, num_agents)
        valid = parent != -2  # (num_samples, num_agents)

        ego_samples = (
            pos - state.env_state.physical_state.agent_pos[None, :, :]
        )  # (num_samples, num_agents, 2)
        dists = jnp.linalg.norm(ego_samples, axis=-1)  # (num_samples, num_agents)
        dists = jnp.where(valid, dists, jnp.inf)
        dists = dists.T  # (num_agents, num_samples)

        nearest_id = jnp.argmin(dists, axis=-1)  # (num_agents, )
        nearest_cost = cost[nearest_id, jnp.arange(self.num_agents)]  # (num_agents, )

        return nearest_cost

    def update_reward(self, reward, state, actions, new_state):
        old_goal_dist = jnp.linalg.norm(state.physical_state.agent_pos - state.goal_pos, axis=-1)
        new_goal_dist = jnp.linalg.norm(new_state.physical_state.agent_pos - new_state.goal_pos, axis=-1)

        clear_reward = reward - self.pos_shaping_factor * (old_goal_dist - new_goal_dist).reshape(
            self.num_agents, 1
        )

        old_cost = self.get_nearest_cost(state)
        new_cost = self.get_nearest_cost(new_state)

        rrt_reward = self.pos_shaping_factor * (old_cost - new_cost).reshape(self.num_agents, 1)

        new_reward = (clear_reward + rrt_reward).reshape(self.num_agents, 1)

        return new_reward

    def get_obs_rrt(self, state):
        pos = state.rrt_state.pos  # (num_samples, num_agents, 2)
        cost = state.rrt_state.cost  # (num_samples, num_agents)
        parent = state.rrt_state.parent  # (num_samples, num_agents)
        valid = parent != -2  # (num_samples, num_agents)
        expanded_window = self.window + self.map_generator.agent_rad

        current_cost = self.get_nearest_cost(state)  # (num_agents, )

        ego_samples = (
            pos - state.env_state.physical_state.agent_pos[None, :, :]
        )  # (num_samples, num_agents, 2)

        dists = jnp.linalg.norm(ego_samples, axis=-1)  # (num_samples, num_agents)
        dists = jnp.where(valid, dists, -jnp.inf)
        dists = jnp.where(dists < expanded_window, dists, -jnp.inf)
        dists = dists.T  # (num_agents, num_samples)

        farthest_dists, farthest_ids = jax.lax.top_k(
            dists, self.num_obs_points
        )  # (num_agents, self.num_obs_points)

        farthest_ego_samples = ego_samples[
            farthest_ids, jnp.arange(self.num_agents)[:, None], :
        ]  # (self.num_agents, self.num_obs_points, 2)
        farthest_costs = cost[
            farthest_ids, jnp.arange(self.num_agents)[:, None]
        ]  # (self.num_agents, self.num_obs_points)
        costs_obs = farthest_costs - current_cost[:, None]  # (self.num_agents, self.num_obs_points)
        costs_obs = costs_obs / (self.window + self.rrt_step_size / 2)

        obs_coeff = expanded_window / farthest_dists  # (num_agents, self.num_obs_points)
        obs = farthest_ego_samples * (1.0 - obs_coeff)[:, :, None]  # (num_agents, self.num_obs_points, 2)
        obs_norm = obs / expanded_window

        obs_norm = jnp.where(
            farthest_dists[:, :, None] < expanded_window, obs_norm, 0.0
        )  # (num_agents, self.num_obs_points, 2)
        obs_norm = jnp.where(
            farthest_dists[:, :, None] == -jnp.inf, 0.0, obs_norm
        )  # (num_agents, self.num_obs_points, 2)
        obs_norm = jnp.where(
            farthest_dists[:, :, None] == 0.0, jnp.array([[1.0, 0.0]]), obs_norm
        )  # (num_agents, self.num_obs_points, 2)

        costs_obs = jnp.where(farthest_dists[:, :] < expanded_window, costs_obs, 0.0)
        costs_obs = jnp.where(farthest_dists[:, :] == -jnp.inf, 0.0, costs_obs)

        is_goal = (
            jnp.linalg.norm(
                farthest_ego_samples
                + state.env_state.physical_state.agent_pos[:, None, :]
                - state.env_state.goal_pos[:, None, :],
                axis=-1,
            )
            < self.map_generator.goal_rad
        )  # (num_agents, self.num_obs_points)
        costs_obs = jnp.where(is_goal, 1.0, costs_obs)

        obs_rrt = jnp.concatenate(
            (obs_norm, costs_obs[:, :, None]), axis=-1
        )  # (num_agents, self.num_obs_points, 3)

        return obs_rrt.reshape(self.num_agents, self.num_obs_points * 3)

    def get_obs(self, state):
        old_obs = self._env.get_obs(state)

        rrt_obs = self.get_obs_rrt(state)

        new_obs = jnp.concatenate(
            (rrt_obs, old_obs), axis=1
        )  # (num_agents, self.num_obs_points * 3 + self.observation_size)

        return new_obs

    def reset(self, key: ArrayLike):
        key_env, key_rrt = jax.random.split(key, 2)
        obs, env_state = self._env.reset(key_env)

        rrt_state = self._rrt.run(
            key_rrt,
            start=env_state.goal_pos,
            goal=env_state.physical_state.agent_pos,
            landmark_pos=env_state.landmark_pos,
        )

        state = State_with_RRTStarState(
            env_state=env_state,
            rrt_state=rrt_state,
        )

        new_obs = self.get_obs(state)

        return new_obs, state

    def step(self, key: ArrayLike, state: State_with_RRTStarState, actions: ArrayLike):
        obs, new_env_state, reward, done, info = self._env.step(key, state.env_state, actions)

        new_state = state.replace(env_state=new_env_state)

        rrt_obs = self.get_obs_rrt(state)

        new_obs = jnp.concatenate(
            (rrt_obs, obs), axis=1
        )  # (num_agents, self.num_obs_points * 3 + self.observation_size)

        new_reward = self.update_reward(reward, state, actions, new_state)

        return new_obs, new_state, new_reward, done, info

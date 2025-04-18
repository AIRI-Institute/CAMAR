from functools import partial
from typing import Tuple, Optional

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from camar.maps import RandomGrid
from camar.maps.base import BaseMap
from camar.utils import Box, State


class Camar:
    def __init__(
        self,
        map_generator: BaseMap = RandomGrid(),
        lifelong: bool = False,
        window: float = 0.8,
        placeholder: float = 0.0,
        max_steps: int = 100,
        frameskip: int = 2,
        max_obs: Optional[int] = None,
        dt: float = 0.01,
        damping: float = 0.25,
        contact_force: float = 500,
        contact_margin: float = 0.001,
        **kwargs,
    ):
        self.device = str(jax.devices()[0])

        self.map_generator = map_generator
        self.window = window

        self.frameskip = frameskip

        self.placeholder = placeholder

        self.height = map_generator.height
        self.width = map_generator.width
        self.landmark_rad = map_generator.landmark_rad
        self.agent_rad = map_generator.agent_rad
        self.goal_rad = map_generator.goal_rad

        if max_obs is not None:
            self.max_obs = max_obs
        else:
            self.max_obs = int(self.window / self.landmark_rad)**2 # for partial observability

        self.max_obs = min(self.max_obs, self.num_entities - 1)

        self.action_size = 2
        self.observation_size = self.max_obs * 2 + 2

        self.action_spaces = Box(low=-1.0, high=1.0, shape=(self.num_agents, self.action_size))
        self.observation_spaces = Box(-jnp.inf, jnp.inf, shape=(self.num_agents, self.observation_size))
        self.action_decoder = self._decode_continuous_action

        # Environment parameters
        self.max_steps = max_steps
        self.dt = dt

        self.mass = kwargs.get("mass", 1.0)
        self.accel = kwargs.get("accel", 5.0)
        self.max_speed = kwargs.get("max_speed", -1)
        self.u_noise = kwargs.get("u_noise", 0)

        self.damping = damping
        self.contact_force = contact_force
        self.contact_margin = contact_margin

        # lifelong
        self.map_reset = map_generator.reset_lifelong if lifelong else map_generator.reset
        self.update_goals = map_generator.update_goals if lifelong else lambda keys, goal_pos, to_update: (keys, goal_pos)

    @property
    def num_agents(self) -> int:
        return self.map_generator.num_agents

    @property
    def num_landmarks(self) -> int:
        return self.map_generator.num_landmarks

    @property
    def num_entities(self) -> int:
        return self.num_agents + self.num_landmarks

    @partial(jax.jit, static_argnums=[0])
    def step(self, key: ArrayLike, state: State, actions: ArrayLike) -> Tuple[State, Array, Array, Array]:
        # actions.shape = (num_agents, 2)
        u = self._decode_continuous_action(actions)

        key, key_w = jax.random.split(key)

        def frameskip(scan_state, _):
            key, state, u = scan_state

            key, _key = jax.random.split(key)
            agent_pos, agent_vel = self._world_step(_key, state, u)

            state = state.replace(
                 agent_pos=agent_pos,
                 agent_vel=agent_vel,
            )
            return (key, state, u), _

        (key, state, u), _ = jax.lax.scan(frameskip, init=(key_w, state, u), xs=None, length=self.frameskip + 1)

        goal_dist = jnp.linalg.norm(state.agent_pos - state.goal_pos, axis=-1) # (num_agents, )
        on_goal = goal_dist < self.goal_rad

        # done = jnp.full((self.num_agents, ), state.step >= self.max_steps)

        done = jnp.logical_or(state.step >= self.max_steps, on_goal.all(axis=-1))

        # terminated = on_goal.all(axis=-1)
        # truncated = state.step >= self.max_steps

        reward = self.get_reward(state.agent_pos, state.landmark_pos, goal_dist)

        goal_keys, goal_pos = self.update_goals(state.goal_keys, state.goal_pos, on_goal)

        state = state.replace(
            goal_pos = goal_pos,
            step = state.step + 1,
            goal_keys = goal_keys,
            on_goal = on_goal,
        )

        obs = self.get_obs(state.agent_pos, state.landmark_pos, state.goal_pos)

        return obs, state, reward, done, {}

    @partial(jax.jit, static_argnums=[0])
    def reset(self, key: ArrayLike) -> Tuple[State, Array, Array]:
        """Initialise with random positions"""

        goal_keys, landmark_pos, agent_pos, goal_pos = self.map_reset(key)

        obs = self.get_obs(agent_pos, landmark_pos, goal_pos)
        # reward = self.get_reward(agent_pos, all_landmark_pos, goal_pos)

        goal_dist = jnp.linalg.norm(agent_pos - goal_pos, axis=-1)
        on_goal = goal_dist < self.goal_rad

        state = State(
            agent_pos = agent_pos,
            agent_vel = jnp.zeros((self.num_agents, 2)),
            goal_pos = goal_pos,
            landmark_pos = landmark_pos,
            on_goal = on_goal,
            step = 0,
            goal_keys = goal_keys,
        )

        return obs, state

    @partial(jax.vmap, in_axes=[None, 0, None])
    def get_dist(self, a_pos: ArrayLike, p_pos: ArrayLike) -> Array:
        return jnp.linalg.norm(a_pos - p_pos, axis=-1)

    @partial(jax.jit, static_argnums=[0])
    def get_obs(self, agent_pos: ArrayLike, landmark_pos: ArrayLike, goal_pos: ArrayLike) -> Array:

        objects = jnp.vstack((agent_pos, landmark_pos)) # (num_objects, 2)

        # (num_agents, 1, 2) - (1, num_objects, 2) -> (num_agents, num_objects, 2) -> (num_agents, num_objects)
        distances = jnp.linalg.norm(agent_pos[:, None, :] - objects[None, :, :], axis=-1)

        nearest_dists, nearest_ids = jax.lax.top_k(-distances, self.max_obs+1) # (num_agents, self.max_obs+1)
        # remove zeros (nearest is the agent itself) -> (num_agents, self.max_obs)
        nearest_ids = nearest_ids[:, 1:]
        nearest_dists = -nearest_dists[:, 1:]

        ego_objects = objects[nearest_ids] - agent_pos[:, None, :] # (num_agents, self.max_obs, 2)

        obs = jnp.where(nearest_dists[:, :, None] < self.window,
                        ego_objects * (1.0 / self.window - 1 / nearest_dists[:, :, None]),
                        self.placeholder)

        ego_goal = goal_pos - agent_pos # [num_agents, 2]

        goal_dist = jnp.linalg.norm(ego_goal, axis=-1)

        ego_goal_clipped = jnp.where(goal_dist[:, None] > 1.0, ego_goal / goal_dist[:, None], ego_goal)

        # ego_goal = - ego_goal

        obs = jnp.concatenate((ego_goal_clipped[:, None, :], obs), axis=1) # (num_agents, self.max_obs + goal, 2)

        return obs.reshape(self.num_agents, self.observation_size)

    def get_reward(self, agent_pos: ArrayLike, landmark_pos: ArrayLike, goal_dist: ArrayLike) -> Array:
        """Return rewards for all agents"""

        objects = jnp.vstack((agent_pos, landmark_pos))

        distances = jnp.linalg.norm(agent_pos[:, None, :] - objects[None, :, :], axis=-1)

        nearest_dists, nearest_ids = jax.lax.top_k(-distances, 2) # (num_agents, 2)

        # remove zeros (nearest is the agent itself) -> (num_agents)
        nearest_ids = nearest_ids[:, 1]
        nearest_dists = -nearest_dists[:, 1]

        effective_rad = jnp.where(nearest_ids < self.num_agents, 2 * self.agent_rad, self.agent_rad + self.landmark_rad)

        collision = nearest_dists < (effective_rad * 1.05)

        on_goal = goal_dist < self.goal_rad

        # r = 10.0 * on_goal.astype(jnp.float32) - 0.001 * goal_dist - 1 * collision.astype(jnp.float32)
        # r = 1.0 * on_goal.astype(jnp.float32) - 0.5 * collision.astype(jnp.float32) - 0.01 * jnp.log1p(goal_dist)
        r = 1.0 * on_goal.astype(jnp.float32) - 2.0 * collision.astype(jnp.float32) - 0.01 * jnp.log(goal_dist + 1e-8)
        # r = 1.0 * on_goal.astype(jnp.float32) - 0.5 * collision.astype(jnp.float32) + jnp.reciprocal(1.0 + goal_dist / 4)
        # r = 1.0 * on_goal.astype(jnp.float32) - 0.5 * collision.astype(jnp.float32)
        return r.reshape(-1, 1)

    def _decode_continuous_action(self, actions: ArrayLike) -> Array:
        """actions (num_agents, 2)"""
        return self.accel * actions

    def _world_step(self, key: ArrayLike, state: State, u: ArrayLike) -> Tuple[Array, Array]:
        # apply agent physical controls
        key_noise = jax.random.split(key, self.num_agents)
        agent_force = self._apply_action_force(key_noise, u)

        # apply environment forces
        agent_force = self._apply_environment_force(agent_force, state)

        # integrate physical state
        agent_pos, agent_vel = self._integrate_state(agent_force, state.agent_pos, state.agent_vel)

        return agent_pos, agent_vel

    # gather agent action forces
    @partial(jax.vmap, in_axes=[None, 0, 0])
    def _apply_action_force(self, key: ArrayLike, u: ArrayLike) -> Array:
        noise = jax.random.normal(key, shape=u.shape) * self.u_noise
        return u + noise

    @partial(jax.vmap, in_axes=[None, 0, 0, 0])
    def _integrate_state(self, force: ArrayLike, pos: ArrayLike, vel:ArrayLike) -> Tuple[Array, Array]:
        """integrate physical state"""

        pos += vel * self.dt
        vel = vel * (1 - self.damping)

        vel += (force / self.mass) * self.dt

        speed = jnp.linalg.norm(vel, ord=2)
        over_max = vel / speed * self.max_speed

        vel = jax.lax.select((speed > self.max_speed) & (self.max_speed >= 0), over_max, vel)

        return pos, vel

    def _apply_environment_force(self, agent_force: ArrayLike, state: State) -> Array:

        # agent - agent
        agent_idx_i, agent_idx_j = jnp.triu_indices(self.num_agents, k=1)
        agent_forces = self._get_collision_force(state.agent_pos[agent_idx_i], state.agent_pos[agent_idx_j], self.agent_rad + self.agent_rad) # (num_agents * (num_agents - 1) / 2, 2)

        agent_force = agent_force.at[agent_idx_i].add(agent_forces)
        agent_force = agent_force.at[agent_idx_j].add(- agent_forces)

        # agent - landmark
        agent_idx = jnp.repeat(jnp.arange(self.num_agents), self.num_landmarks)
        landmark_idx = jnp.tile(jnp.arange(self.num_landmarks), self.num_agents)
        landmark_forces = self._get_collision_force(state.agent_pos[agent_idx], state.landmark_pos[landmark_idx], self.agent_rad + self.landmark_rad) # (num_agents * num_landmarks, 2)

        agent_force = agent_force.at[agent_idx].add(landmark_forces)

        return agent_force

    @partial(jax.vmap, in_axes=[None, 0, 0, None])
    def _get_collision_force(self, pos_a: ArrayLike, pos_b: ArrayLike, min_dist: float) -> Array:
        delta_pos = pos_a - pos_b

        dist = jnp.linalg.norm(delta_pos, axis=-1)

        # softmax penetration
        k = self.contact_margin
        penetration = jnp.logaddexp(0, - (dist - min_dist) / k) * k
        force = self.contact_force * delta_pos / jax.lax.select(dist > 0, dist, jnp.full(dist.shape, 1e-8)) * penetration

        return force

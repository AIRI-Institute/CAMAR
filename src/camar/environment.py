from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from camar.maps import random_grid
from camar.maps.base_map import base_map
from camar.utils import Box, State


class Camar:
    def __init__(
        self,
        map_generator: base_map = random_grid(),
        lifelong: bool = False,
        window: Optional[float] = None,
        placeholder: float = 0.0,
        max_steps: int = 100,
        frameskip: int = 1,
        max_obs: Optional[int] = None,
        pos_shaping_factor: Optional[float] = None,
        dt: float = 0.01,
        damping: float = 0.25,
        contact_force: float = 500,
        contact_margin: float = 0.001,
        **kwargs,
    ):
        self.device = str(jax.devices()[0])

        self.map_generator = map_generator

        self.frameskip = frameskip

        self.placeholder = placeholder

        self.height = map_generator.height
        self.width = map_generator.width
        self.landmark_rad = map_generator.landmark_rad
        self.agent_rad = map_generator.agent_rad
        self.goal_rad = map_generator.goal_rad

        if pos_shaping_factor is None:
            self.pos_shaping_factor = 0.04 / self.agent_rad
        else:
            self.pos_shaping_factor = pos_shaping_factor

        if window is None:
            self.window = max(self.landmark_rad, self.agent_rad) * 2.5
        else:
            self.window = window

        if max_obs is not None:
            self.max_obs = max_obs
        else:
            self.max_obs = int(self.window / self.landmark_rad)**2 # for partial observability

        self.max_obs = min(self.max_obs, self.num_entities - 1)

        self.action_size = 2
        self.observation_size = self.max_obs * 2 + 2

        self.action_spaces = Box(low=-1.0, high=1.0, shape=(self.num_agents, self.action_size))
        self.observation_spaces = Box(-jnp.inf, jnp.inf, shape=(self.num_agents, self.observation_size))

        # Environment parameters
        self.max_steps = max_steps
        self.dt = dt
        self.step_dt = self.dt * (self.frameskip + 1) # for metrics
        self.max_time = self.max_steps * self.step_dt # for metrics

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

    def step(self, key: ArrayLike, state: State, actions: ArrayLike) -> tuple[Array, State, Array, Array, dict]:
        # actions = (num_agents, 2)
        u = self.accel * actions

        key, key_w = jax.random.split(key)

        old_goal_dist = jnp.linalg.norm(state.agent_pos - state.goal_pos, axis=-1) # (num_agents, )

        def frameskip(scan_state, _):
            key, state, u, is_collision = scan_state

            key, _key = jax.random.split(key)
            state, is_collision = self._world_step(_key, state, u, is_collision)

            return (key, state, u, is_collision), None

        is_collision = jnp.zeros(shape=(self.num_agents, ), dtype=jnp.int32)

        (key, state, u, is_collision), _ = jax.lax.scan(frameskip, init=(key_w, state, u, is_collision), xs=None, length=self.frameskip + 1)

        is_collision = is_collision >= 1

        goal_dist = jnp.linalg.norm(state.agent_pos - state.goal_pos, axis=-1) # (num_agents, )
        on_goal = goal_dist < self.goal_rad

        # done = jnp.full((self.num_agents, ), state.step >= self.max_steps)

        done = jnp.logical_or(state.step >= self.max_steps, on_goal.all(axis=-1))

        # terminated = on_goal.all(axis=-1)
        # truncated = state.step >= self.max_steps

        reward = self.get_reward(is_collision, goal_dist, old_goal_dist)

        goal_keys, goal_pos = self.update_goals(state.goal_keys, state.goal_pos, on_goal)

        just_arrived = jnp.logical_not(state.on_goal) & on_goal
        current_time = (state.step + 1) * self.step_dt
        time_to_reach_goal = jnp.where(just_arrived, current_time, state.time_to_reach_goal)
        num_collisions = state.num_collisions + is_collision.astype(jnp.int32)

        state = state.replace(
            goal_pos = goal_pos,
            is_collision = is_collision,
            step = state.step + 1,
            goal_keys = goal_keys,
            on_goal = on_goal,
            time_to_reach_goal = time_to_reach_goal,
            num_collisions = num_collisions,
        )

        obs = self.get_obs(state)

        return obs, state, reward, done, {}

    def reset(self, key: ArrayLike) -> tuple[Array, State]:
        """Initialise with random positions"""

        goal_keys, landmark_pos, agent_pos, goal_pos = self.map_reset(key)

        # reward = self.get_reward(agent_pos, all_landmark_pos, goal_pos)

        goal_dist = jnp.linalg.norm(agent_pos - goal_pos, axis=-1)
        on_goal = goal_dist < self.goal_rad

        state = State(
            agent_pos = agent_pos,
            agent_vel = jnp.zeros((self.num_agents, 2)),
            goal_pos = goal_pos,
            landmark_pos = landmark_pos,
            is_collision = jnp.full((self.num_agents, ), False, dtype=jnp.bool_),
            step = 0,
            on_goal = on_goal,
            time_to_reach_goal = jnp.full((self.num_agents, ), self.max_time),
            num_collisions = jnp.zeros((self.num_agents, ), dtype=jnp.int32),
            goal_keys = goal_keys,
        )

        obs = self.get_obs(state)

        return obs, state

    @partial(jax.vmap, in_axes=[None, 0, None])
    def get_dist(self, a_pos: ArrayLike, p_pos: ArrayLike) -> Array:
        return jnp.linalg.norm(a_pos - p_pos, axis=-1)

    def get_obs(self, state: State) -> Array:
        agent_pos = state.agent_pos
        goal_pos = state.goal_pos
        landmark_pos = state.landmark_pos

        objects = jnp.vstack((agent_pos, landmark_pos)) # (num_objects, 2)

        # (1, num_objects, 2) - (num_agents, 1, 2) -> (num_agents, num_objecst, 2)
        ego_objects = objects[None, :, :] - agent_pos[:, None, :]

        # (num_agents, num_objecst, 2) -> (num_agents, num_objecst)
        dists = jnp.linalg.norm(ego_objects, axis=-1)
        nearest_dists, nearest_ids = jax.lax.top_k(- dists, self.max_obs + 1) # (num_agents, self.max_obs + 1)
        # remove zero dists (nearest is the agent itself) -> (num_agents, self.max_obs)
        nearest_ids = nearest_ids[:, 1:]
        nearest_dists = -nearest_dists[:, 1:]

        nearest_ego_objects = ego_objects[jnp.arange(self.num_agents)[:, None], nearest_ids] # (num_agents, self.max_obs, 2)
        nearest_rad = jnp.where(nearest_ids < self.num_agents,
                                self.agent_rad,
                                self.landmark_rad) # (num_agents, self.max_obs)

        obs_dists_coeff = (self.agent_rad + nearest_rad) / nearest_dists # (num_agents, self.max_obs)
        obs_dists = jnp.linalg.norm(nearest_ego_objects * (1.0 - obs_dists_coeff)[:, :, None], axis=-1)  # (num_agents, self.max_obs)

        obs_coeff = (self.window + nearest_rad) / nearest_dists # (num_agents, self.max_obs)
        obs = nearest_ego_objects * (1.0 - obs_coeff)[:, :, None] # (num_agents, self.max_obs, 2)
        obs_norm = obs / (self.window - self.agent_rad)

        obs_norm = jnp.where(obs_dists[:, :, None] < self.window,
                             obs_norm,
                             0.0)  # (num_agents, self.max_obs, 2)

        ego_goal = goal_pos - agent_pos # [num_agents, 2]

        goal_dist = jnp.linalg.norm(ego_goal, axis=-1)

        ego_goal_norm = jnp.where(goal_dist[:, None] > 1.0,
                                  ego_goal / goal_dist[:, None],
                                  ego_goal)

        obs = jnp.concatenate((ego_goal_norm[:, None, :], obs_norm), axis=1) # (num_agents, self.max_obs + goal, 2)

        return obs.reshape(self.num_agents, self.observation_size)

    def get_reward(self, is_collision: ArrayLike, goal_dist: ArrayLike, old_goal_dist: ArrayLike) -> Array:

        on_goal = goal_dist < self.goal_rad

        r = 0.5 * on_goal.astype(jnp.float32) - 1.0 * is_collision.astype(jnp.float32) + self.pos_shaping_factor * (old_goal_dist - goal_dist)
        return r.reshape(-1, 1)

    def _world_step(self, key: ArrayLike, state: State, u: ArrayLike, is_collision: ArrayLike) -> tuple[State, Array]:

        agent_force = self._add_noise(key, u)

        # apply collision forces
        agent_force, is_collision = self._apply_environment_force(agent_force, is_collision, state)

        # integrate state
        state = self._integrate_state(agent_force, state)

        return state, is_collision

    def _add_noise(self, key: ArrayLike, u: ArrayLike) -> Array:
        noise = jax.random.normal(key, shape=u.shape) * self.u_noise
        return u + noise

    def _integrate_state(self, force: ArrayLike, state: State) -> State:
        """integrate physical state"""
        pos = state.agent_pos
        vel = state.agent_vel

        pos += vel * self.dt
        vel = vel * (1 - self.damping)

        vel += (force / self.mass) * self.dt

        speed = jnp.linalg.norm(vel, axis=-1, keepdims=True)
        over_max = vel / speed * self.max_speed

        vel = jnp.where((speed > self.max_speed) & (self.max_speed >= 0), over_max, vel)

        state = state.replace(
            agent_pos = pos,
            agent_vel = vel,
        )

        return state

    def _apply_environment_force(self, agent_force: ArrayLike, is_collision: ArrayLike, state: State) -> tuple[Array, Array]:

        # agent - agent collisions
        agent_idx_i, agent_idx_j = jnp.triu_indices(self.num_agents, k=1)
        agent_forces, is_collision_agents = self._get_collision_force(state.agent_pos[agent_idx_i], state.agent_pos[agent_idx_j], self.agent_rad + self.agent_rad) # (num_agents * (num_agents - 1) / 2, 2)

        is_collision = is_collision.at[agent_idx_i].add(is_collision_agents)
        is_collision = is_collision.at[agent_idx_j].add(is_collision_agents)

        agent_force = agent_force.at[agent_idx_i].add(agent_forces)
        agent_force = agent_force.at[agent_idx_j].add(- agent_forces)

        # agent - landmark collisions
        agent_idx = jnp.repeat(jnp.arange(self.num_agents), self.num_landmarks)
        landmark_idx = jnp.tile(jnp.arange(self.num_landmarks), self.num_agents)
        landmark_forces, is_collision_landmarks = self._get_collision_force(state.agent_pos[agent_idx], state.landmark_pos[landmark_idx], self.agent_rad + self.landmark_rad) # (num_agents * num_landmarks, 2)

        is_collision = is_collision.at[agent_idx].add(is_collision_landmarks)

        agent_force = agent_force.at[agent_idx].add(landmark_forces)

        return agent_force, is_collision

    @partial(jax.vmap, in_axes=[None, 0, 0, None])
    def _get_collision_force(self, pos_a: ArrayLike, pos_b: ArrayLike, min_dist: float) -> tuple[Array, Array]:
        delta_pos = pos_a - pos_b

        dist = jnp.linalg.norm(delta_pos, axis=-1)

        # softmax penetration
        k = self.contact_margin
        penetration = jnp.logaddexp(0, - (dist - min_dist) / k) * k
        force = self.contact_force * delta_pos / jax.lax.select(dist > 0, dist, jnp.full(dist.shape, 1e-8)) * penetration
        is_collision = (dist < min_dist).astype(jnp.int32)
        return force, is_collision

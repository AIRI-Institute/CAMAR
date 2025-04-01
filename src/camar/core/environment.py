from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from camar.core.maps.base import Map
from camar.core.utils import Box, State


class Env:
    def __init__(
        self,
        width: int = 19,
        height: int = 19,
        obstacle_density: float = 0.2,
        num_agents: int = 8,
        grain_factor: int = 4,
        obstacle_size: float = 0.4,
        goal_rad: float = 0.04,
        window_rad: int = 2,
        placeholder: float = 0.0,
        max_steps: int = 100,
        dt: float = 0.01,
        damping: float = 0.25,
        contact_force: float = 500,
        contact_margin: float = 0.001,
        frameskip: int = 5,
        max_obs: int | None = None,
        **kwargs,
    ):
        self.grain_factor = grain_factor
        self.obstacle_size = obstacle_size
        self.goal_rad = goal_rad
        self.window = obstacle_size * window_rad
        self.num_agents = num_agents
        self.width = width
        self.height = height

        self.frameskip = frameskip

        self.num_obstacles = int(obstacle_density * width * height)

        self.placeholder = placeholder
        self.obs_placeholder = jnp.full(shape=(self.num_obstacles, 2), fill_value=placeholder)

        self.num_landmarks = self.num_obstacles * 4 * (self.grain_factor - 1) + (width + height) * 2 * (self.grain_factor - 1)
        self.num_entities = self.num_agents + self.num_landmarks

        self.agent_range = jnp.arange(0, self.num_agents)
        self.landmark_range = jnp.arange(self.num_agents, self.num_entities)
        self.entity_range = jnp.arange(0, self.num_entities)

        half_width = width * self.obstacle_size / 2
        half_height = height * self.obstacle_size / 2

        x_coords = jnp.linspace(
            - half_width + self.obstacle_size / 2, # start
            half_width - self.obstacle_size / 2, # end
            width # map width
        )
        y_coords = jnp.linspace(
            - half_height + self.obstacle_size / 2, # start
            half_height - self.obstacle_size / 2, # end
            height # map height
        )

        self.map_coordinates = jnp.stack(jnp.meshgrid(x_coords, y_coords), axis=-1).reshape(-1, 2)
        self.border_landmarks = self.get_border_landmarks(width, height, half_width, half_height, self.grain_factor)

        self.landmark_rad = self.obstacle_size / (2 * (self.grain_factor - 1))
        self.agent_rad = (self.obstacle_size - 2 * self.landmark_rad) * 0.4

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
        
        self.colour = [(115, 243, 115) for i in jnp.arange(self.num_agents)] + [(64, 64, 64) for i in jnp.arange(self.num_landmarks)]

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
            axis=-1
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
            axis=-1
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
            axis=-1
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
            axis=-1
        )
        return jnp.concatenate([top_wall, right_wall, left_wall, bottom_wall])

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

        reward = self.rewards(state.agent_pos, state.landmark_pos, goal_dist)

        obs = self.get_obs(state.agent_pos, state.landmark_pos, state.goal_pos)

        state = state.replace(
            step=state.step + 1,
        )

        return state, obs, reward, done

    @partial(jax.jit, static_argnums=[0])
    def reset(self, key: ArrayLike) -> Tuple[State, Array, Array]:
        """Initialise with random positions"""
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

        @partial(jax.vmap, in_axes=[0, None, None], out_axes=1)
        def get_landmarks(obstacle, grain_factor, obstacle_size):
            left_x, down_y = obstacle - obstacle_size / 2
            right_x, up_y = obstacle + obstacle_size / 2
            
            up_landmarks = jnp.stack((jnp.linspace(left_x, right_x, grain_factor - 1, endpoint=False), jnp.full((grain_factor - 1, ), up_y)), axis=-1)
            right_landmarks = jnp.stack((jnp.full((grain_factor - 1, ), right_x), jnp.linspace(up_y, down_y, grain_factor - 1, endpoint=False)), axis=-1)
            down_landmarks = jnp.stack((jnp.linspace(right_x, left_x, grain_factor - 1, endpoint=False), jnp.full((grain_factor - 1, ), down_y)), axis=-1)
            left_landmarks = jnp.stack((jnp.full((grain_factor - 1, ), left_x), jnp.linspace(down_y, up_y, grain_factor - 1, endpoint=False)), axis=-1)

            return jnp.concatenate([up_landmarks, right_landmarks, down_landmarks, left_landmarks])
        
        landmark_pos = get_landmarks(obstacle_pos, self.grain_factor, self.obstacle_size).reshape(-1, 2)

        all_landmark_pos = jnp.concatenate(
            [
                landmark_pos,
                self.border_landmarks,
            ]
        )

        obs = self.get_obs(agent_pos, all_landmark_pos, goal_pos)
        # reward = self.rewards(agent_pos, all_landmark_pos, goal_pos)

        state = State(
            agent_pos=agent_pos,
            agent_vel=jnp.zeros((self.num_agents, 2)),
            goal_pos=goal_pos,
            # obstacle_pos=obstacle_pos,
            landmark_pos=all_landmark_pos,
            # observation=obs,
            # reward=reward,
            # done=jnp.full((self.num_agents), False),
            # done=jnp.array([False]),
            step=0,
        )

        goal_dist = jnp.linalg.norm(state.agent_pos - state.goal_pos, axis=-1)
        on_goal = goal_dist < self.goal_rad
        done = on_goal.all(axis=-1)

        return state, obs, done
    
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

    def rewards(self, agent_pos: ArrayLike, landmark_pos: ArrayLike, goal_dist: ArrayLike) -> Array:
        """Assign rewards for all agents"""

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
        """
        actions (num_agents, 2)
        """
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

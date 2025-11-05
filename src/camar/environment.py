import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from camar.dynamics import BaseDynamic, HolonomicDynamic, PhysicalState
from camar.maps import random_grid
from camar.maps.base import base_map
from camar.utils import Box, State


class Camar:
    def __init__(
        self,
        map_generator: base_map = random_grid(),
        dynamic: BaseDynamic = HolonomicDynamic(),
        lifelong: bool = False,  # TODO: not supported yet
        window: float = 0.3,
        max_steps: int = 100,
        frameskip: int = 1,
        max_obs: int = 3,
        pos_shaping_factor: float = 0.0,
        contact_force: float = 500,
        contact_margin: float = 0.001,
    ):
        self.map_generator = map_generator
        assert isinstance(map_generator, base_map), "map_generator must be an instance of base_map"

        self.dynamic = dynamic
        assert isinstance(dynamic, BaseDynamic), "dynamic must be an instance of BaseDynamic"
        # TODO: this is incompatible with MixedDynamic due to frozen logic of flax.struct.dataclass
        # assert isinstance(dynamic.state_class, type(PhysicalState)), (
        #     "dynamic.state_class must be a subclass of PhysicalState"
        # )

        if hasattr(dynamic, "num_agents"):
            if dynamic.num_agents != map_generator.num_agents:
                raise ValueError(f"{dynamic.num_agents=} must be equal to {map_generator.num_agents=}")
            else:
                print(f"{dynamic.num_agents=} has been matched with {map_generator.num_agents=}")

        self.frameskip = frameskip
        self.pos_shaping_factor = pos_shaping_factor
        self.window = window
        self.max_obs = min(max_obs, self.num_agents + self.num_landmarks - 1)

        self.action_spaces = Box(
            low=-1.0, high=1.0, shape=(self.num_agents, self.action_size)
        )  # always [-1; 1] - dynamic changes the range
        self.observation_spaces = Box(-jnp.inf, jnp.inf, shape=(self.num_agents, self.observation_size))

        self.max_steps = max_steps
        self.step_dt = self.dt * (self.frameskip + 1)
        self.max_time = self.max_steps * self.step_dt

        self.contact_force = contact_force
        self.contact_margin = contact_margin

        # lifelong
        self.map_reset = map_generator.reset_lifelong if lifelong else map_generator.reset
        self.update_goals = (
            map_generator.update_goals if lifelong else lambda keys, goal_pos, to_update: (keys, goal_pos)
        )

    @property
    def homogeneous_agents(self) -> bool:
        return self.map_generator.homogeneous_agents

    @property
    def homogeneous_landmarks(self) -> bool:
        return self.map_generator.homogeneous_landmarks

    @property
    def homogeneous_goals(self) -> bool:
        return self.map_generator.homogeneous_goals

    @property
    def num_agents(self) -> int:
        return self.map_generator.num_agents

    @property
    def num_landmarks(self) -> int:
        return self.map_generator.num_landmarks

    @property
    def height(self) -> int:
        return self.map_generator.height

    @property
    def width(self) -> int:
        return self.map_generator.width

    @property
    def action_size(self) -> int:
        return self.dynamic.action_size

    @property
    def dt(self) -> float:
        return self.dynamic.dt

    @property
    def observation_size(self) -> int:
        return self.max_obs * 2 + 2

    def reset(self, key: ArrayLike) -> tuple[Array, State]:
        goal_keys, landmark_pos, agent_pos, goal_pos, sizes = self.map_reset(key)

        goal_dist = jnp.linalg.norm(agent_pos - goal_pos, axis=-1)
        on_goal = goal_dist < (self.map_generator.goal_rad if self.homogeneous_goals else sizes.goal_rad)

        physical_state = self.dynamic.state_class.create(key, landmark_pos, agent_pos, goal_pos, sizes)

        state = State(
            physical_state=physical_state,
            landmark_pos=landmark_pos,
            goal_pos=goal_pos,
            sizes=sizes,
            is_collision=jnp.full((self.num_agents,), False, dtype=jnp.bool_),  # no checks on reset
            step=0,
            on_goal=on_goal,
            time_to_reach_goal=jnp.full((self.num_agents,), self.max_time),
            num_collisions=jnp.zeros((self.num_agents,), dtype=jnp.int32),
            goal_keys=goal_keys,
        )

        obs = self.get_obs(state)

        return obs, state

    def step(
        self, key: ArrayLike, state: State, actions: ArrayLike
    ) -> tuple[Array, State, Array, Array, dict]:
        # actions (num_agents, 2)
        key, key_w = jax.random.split(key)

        def _frameskip(scan_state, _):
            key, physical_state, landmark_pos, actions, sizes, is_collision = scan_state

            key, _key = jax.random.split(key)
            physical_state, is_collision = self._world_step(
                _key, physical_state, actions, landmark_pos, sizes, is_collision
            )

            return (key, physical_state, landmark_pos, actions, sizes, is_collision), None

        is_collision = jnp.zeros(shape=(self.num_agents,), dtype=jnp.int32)

        scan_state = (
            key_w,
            state.physical_state,
            state.landmark_pos,
            actions,
            state.sizes,
            is_collision,
        )

        scan_state, _ = jax.lax.scan(
            _frameskip,
            init=scan_state,
            xs=None,
            length=self.frameskip + 1,
        )
        key_w, physical_state, landmark_pos, actions, sizes, is_collision = scan_state

        is_collision = is_collision >= 1

        goal_dist = jnp.linalg.norm(physical_state.agent_pos - state.goal_pos, axis=-1)  # (num_agents, )
        on_goal = goal_dist < (
            self.map_generator.goal_rad if self.homogeneous_goals else state.sizes.goal_rad
        )  # (num_agents, )

        # done = jnp.full((self.num_agents, ), state.step >= self.max_steps)
        done = jnp.logical_or(state.step >= self.max_steps, on_goal.all(axis=-1))

        # terminated = on_goal.all(axis=-1)
        # truncated = state.step >= self.max_steps

        # reward = self.get_reward(is_collision, goal_dist, old_goal_dist)

        goal_keys, goal_pos = self.update_goals(state.goal_keys, state.goal_pos, on_goal)

        just_arrived = jnp.logical_not(state.on_goal) & on_goal
        current_time = (state.step + 1) * self.step_dt
        time_to_reach_goal = jnp.where(just_arrived, current_time, state.time_to_reach_goal)
        num_collisions = state.num_collisions + is_collision.astype(jnp.int32)

        new_state = state.replace(
            physical_state=physical_state,
            goal_pos=goal_pos,
            sizes=sizes,
            is_collision=is_collision,
            step=state.step + 1,
            on_goal=on_goal,
            time_to_reach_goal=time_to_reach_goal,
            num_collisions=num_collisions,
            goal_keys=goal_keys,
        )

        obs = self.get_obs(new_state)
        reward = self.get_reward(state, actions, new_state)

        return obs, new_state, reward, done, {}

    def get_obs(self, state: State) -> Array:
        agent_pos = state.physical_state.agent_pos
        landmark_pos = state.landmark_pos

        goal_pos = state.goal_pos

        objects = jnp.vstack((agent_pos, landmark_pos))  # (num_objects, 2)

        # (1, num_objects, 2) - (num_agents, 1, 2) -> (num_agents, num_objecst, 2)
        ego_objects = objects[None, :, :] - agent_pos[:, None, :]

        # (num_agents, num_objecst, 2) -> (num_agents, num_objecst)
        dists = jnp.linalg.norm(ego_objects, axis=-1)
        nearest_dists, nearest_ids = jax.lax.top_k(-dists, self.max_obs + 1)  # (num_agents, self.max_obs + 1)
        # remove zero dists (nearest is the agent itself) -> (num_agents, self.max_obs)
        nearest_ids = nearest_ids[:, 1:]
        nearest_dists = -nearest_dists[:, 1:]

        nearest_ego_objects = ego_objects[
            jnp.arange(self.num_agents)[:, None], nearest_ids
        ]  # (num_agents, self.max_obs, 2)
        nearest_rad = jnp.where(
            nearest_ids < self.num_agents,
            self.map_generator.agent_rad if self.homogeneous_agents else state.sizes.agent_rad[:, None],
            self.map_generator.landmark_rad
            if self.homogeneous_landmarks
            else state.sizes.landmark_rad[:, None],
        )  # (num_agents, self.max_obs)

        obs_dists_coeff = (
            (self.map_generator.agent_rad if self.homogeneous_agents else state.sizes.agent_rad[:, None])
            + nearest_rad
        ) / nearest_dists  # (num_agents, self.max_obs)
        obs_dists = jnp.linalg.norm(
            nearest_ego_objects * (1.0 - obs_dists_coeff)[:, :, None], axis=-1
        )  # (num_agents, self.max_obs)

        obs_coeff = (self.window + nearest_rad) / nearest_dists  # (num_agents, self.max_obs)
        obs = nearest_ego_objects * (1.0 - obs_coeff)[:, :, None]  # (num_agents, self.max_obs, 2)
        obs_norm = obs / (
            self.window
            - (
                self.map_generator.agent_rad
                if self.homogeneous_agents
                else state.sizes.agent_rad[:, None, None]
            )
        )

        obs_norm = jnp.where(
            obs_dists[:, :, None] < self.window, obs_norm, 0.0
        )  # (num_agents, self.max_obs, 2)

        ego_goal = goal_pos - agent_pos  # [num_agents, 2]

        goal_dist = jnp.linalg.norm(ego_goal, axis=-1)

        ego_goal_norm = jnp.where(
            goal_dist[:, None] > self.window, ego_goal / goal_dist[:, None], ego_goal / self.window
        )

        obs = jnp.concatenate(
            (ego_goal_norm[:, None, :], obs_norm), axis=1
        )  # (num_agents, self.max_obs + goal, 2)

        return obs.reshape(self.num_agents, self.observation_size)

    def get_reward(self, state: State, actions: ArrayLike, new_state: State) -> Array:
        old_goal_dist = jnp.linalg.norm(state.physical_state.agent_pos - state.goal_pos, axis=-1)
        new_goal_dist = jnp.linalg.norm(new_state.physical_state.agent_pos - new_state.goal_pos, axis=-1)
        on_goal = new_goal_dist < (
            self.map_generator.goal_rad if self.homogeneous_goals else new_state.sizes.goal_rad
        )

        r = (
            +0.5 * on_goal.astype(jnp.float32)
            + 0.5 * on_goal.all(axis=-1).astype(jnp.float32)
            - 1.0 * new_state.is_collision.astype(jnp.float32)
            + self.pos_shaping_factor * (old_goal_dist - new_goal_dist)
        )
        return r.reshape(-1, 1)

    def _world_step(
        self,
        key: ArrayLike,
        physical_state: PhysicalState,
        actions: ArrayLike,
        landmark_pos: ArrayLike,
        sizes: ArrayLike,
        is_collision: ArrayLike,
    ) -> tuple[PhysicalState, Array]:
        collision_force = jnp.zeros((self.num_agents, 2))

        # apply collision forces
        collision_force, is_collision = self._get_environment_force(
            collision_force,
            physical_state,
            landmark_pos,
            sizes,
            is_collision,
        )

        # integrate state
        physical_state = self.dynamic.integrate(key, collision_force, physical_state, actions)

        return physical_state, is_collision

    def _get_environment_force(
        self,
        agent_force: ArrayLike,
        physical_state: PhysicalState,
        landmark_pos: ArrayLike,
        sizes: ArrayLike,
        is_collision: ArrayLike,
    ) -> tuple[Array, Array]:
        agent_rad = self.map_generator.agent_rad if self.homogeneous_agents else sizes.agent_rad
        landmark_rad = self.map_generator.landmark_rad if self.homogeneous_landmarks else sizes.landmark_rad

        # agent vs agent collisions
        agent_idx_i, agent_idx_j = jnp.triu_indices(self.num_agents, k=1)
        agent_forces, is_collision_agents = self._calculate_collision_force(
            physical_state.agent_pos[agent_idx_i],
            physical_state.agent_pos[agent_idx_j],
            2 * agent_rad if self.homogeneous_agents else agent_rad[agent_idx_i] + agent_rad[agent_idx_j],
        )  # (num_agents * (num_agents - 1) / 2, 2)

        is_collision = is_collision.at[agent_idx_i].add(is_collision_agents)
        is_collision = is_collision.at[agent_idx_j].add(is_collision_agents)

        agent_force = agent_force.at[agent_idx_i].add(agent_forces)
        agent_force = agent_force.at[agent_idx_j].add(-agent_forces)

        # agent vs landmark collisions
        agent_idx = jnp.repeat(jnp.arange(self.num_agents), self.num_landmarks)
        landmark_idx = jnp.tile(jnp.arange(self.num_landmarks), self.num_agents)
        landmark_forces, is_collision_landmarks = self._calculate_collision_force(
            physical_state.agent_pos[agent_idx],
            landmark_pos[landmark_idx],
            (agent_rad if self.homogeneous_agents else agent_rad[agent_idx])
            + (landmark_rad if self.homogeneous_landmarks else landmark_rad[landmark_idx]),
        )  # (num_agents * num_landmarks, 2)

        is_collision = is_collision.at[agent_idx].add(is_collision_landmarks)

        agent_force = agent_force.at[agent_idx].add(landmark_forces)

        return agent_force, is_collision

    def _calculate_collision_force(
        self, pos_a: ArrayLike, pos_b: ArrayLike, min_dist: float | ArrayLike
    ) -> tuple[Array, Array]:
        delta_pos = pos_a - pos_b

        dist = jnp.linalg.norm(delta_pos, axis=-1)

        # not bool because there is no array.at[idx].or()
        is_collision = (dist <= min_dist).astype(jnp.int32)

        penetration = (
            jnp.logaddexp(0, -(dist - min_dist) / self.contact_margin) * self.contact_margin
        ).reshape(-1, 1)

        dist = dist.reshape(-1, 1)
        force = (
            self.contact_force
            * delta_pos
            / jax.lax.select(dist > 0, dist, jnp.full(dist.shape, 1e-8))
            * penetration
        )
        force = jnp.where(is_collision.reshape(-1, 1), force, jnp.zeros_like(force))

        return force, is_collision

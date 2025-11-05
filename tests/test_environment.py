import jax
import jax.numpy as jnp

from camar import camar_v0
from camar.environment import Camar
from camar.maps import random_grid, string_grid
from camar.dynamics import HolonomicDynamic, DiffDriveDynamic


class TestEnvironment:
    def test_env_creation(self):
        env = camar_v0()
        assert isinstance(env, Camar)

        map_gen = random_grid(num_agents=4)
        env = camar_v0(map_generator=map_gen)
        assert isinstance(env, Camar)
        assert env.num_agents == 4

    def test_env_reset(self):
        env = camar_v0()
        key = jax.random.key(0)
        obs, state = env.reset(key)

        assert obs is not None
        assert state is not None
        assert obs.shape == (env.num_agents, env.observation_size)
        assert state.physical_state.agent_pos.shape == (env.num_agents, 2)

    def test_env_step(self):
        env = camar_v0(map_generator=random_grid(num_agents=2))
        key = jax.random.key(0)
        key_r, key_a, key_s = jax.random.split(key, 3)

        obs, state = env.reset(key_r)
        actions = env.action_spaces.sample(key_a)

        step_fn = jax.jit(env.step)
        obs_next, state_next, reward, done, info = step_fn(key_s, state, actions)

        assert obs_next is not None
        assert state_next is not None
        assert reward is not None
        assert done is not None
        assert info is not None

        assert obs_next.shape == (env.num_agents, env.observation_size)
        assert reward.shape == (env.num_agents, 1)
        assert isinstance(done, jnp.ndarray)

    def test_env_step_with_string_grid(self):
        map_str = """
        .....#.....
        .....#.....
        ...........
        .....#.....
        .....#.....
        """
        env = camar_v0(map_generator=string_grid(map_str=map_str, num_agents=2))
        key = jax.random.key(0)
        key_r, key_a, key_s = jax.random.split(key, 3)

        obs, state = env.reset(key_r)
        actions = env.action_spaces.sample(key_a)

        obs_next, state_next, reward, done, info = env.step(key_s, state, actions)

        assert obs_next is not None
        assert state_next is not None
        assert reward is not None
        assert done is not None
        assert info is not None

    def test_batched_env_reset_and_step(self):
        num_envs = 4
        env = camar_v0(map_generator=random_grid(num_agents=2))
        key = jax.random.key(0)
        key_r, key_a, key_s = jax.random.split(key, 3)

        env_reset_fn = jax.jit(jax.vmap(env.reset))
        env_step_fn = jax.jit(jax.vmap(env.step))
        action_sampler = jax.jit(jax.vmap(env.action_spaces.sample))

        keys_r = jax.random.split(key_r, num_envs)
        obs, state = env_reset_fn(keys_r)

        assert obs.shape == (num_envs, env.num_agents, env.observation_size)

        keys_a = jax.random.split(key_a, num_envs)
        actions = action_sampler(keys_a)

        keys_s = jax.random.split(key_s, num_envs)
        obs_next, state_next, reward, done, info = env_step_fn(keys_s, state, actions)

        assert obs_next.shape == (num_envs, env.num_agents, env.observation_size)
        assert reward.shape == (num_envs, env.num_agents, 1)
        assert info is not None

    def test_env_with_holonomic_dynamic(self):
        env = camar_v0(
            map_generator=random_grid(num_agents=2),
            dynamic=HolonomicDynamic(accel=10.0, max_speed=8.0, damping=0.5),
        )
        key = jax.random.key(0)
        key_r, key_a, key_s = jax.random.split(key, 3)

        obs, state = env.reset(key_r)
        actions = env.action_spaces.sample(key_a)

        obs_next, state_next, reward, done, info = env.step(key_s, state, actions)

        assert type(env.dynamic) is HolonomicDynamic
        assert env.dynamic.accel == 10.0
        assert env.dynamic.max_speed == 8.0
        assert env.dynamic.damping == 0.5
        assert obs_next.shape == (env.num_agents, env.observation_size)
        assert reward.shape == (env.num_agents, 1)

    def test_env_with_diffdrive_dynamic(self):
        env = camar_v0(
            map_generator=string_grid(map_str="...", num_agents=1),
            dynamic=DiffDriveDynamic(linear_speed_max=2.0, angular_speed_max=3.0, mass=2.0),
        )
        key = jax.random.key(0)
        key_r, key_a, key_s = jax.random.split(key, 3)

        obs, state = env.reset(key_r)
        actions = env.action_spaces.sample(key_a)

        obs_next, state_next, reward, done, info = env.step(key_s, state, actions)

        assert type(env.dynamic) is DiffDriveDynamic
        assert env.dynamic.linear_speed_max == 2.0
        assert env.dynamic.angular_speed_max == 3.0
        assert env.dynamic.mass == 2.0
        assert obs_next.shape == (env.num_agents, env.observation_size)
        assert reward.shape == (env.num_agents, 1)

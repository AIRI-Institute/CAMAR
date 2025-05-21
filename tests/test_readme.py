import jax

from camar import camar_v0
from camar.maps import labmaze_grid, random_grid, string_grid
from camar.wrappers import OptimisticResetVecEnvWrapper


class TestReadmeExamples:
    def test_quickstart_single_env(self):
        key = jax.random.key(0)
        key, key_r, key_a, key_s = jax.random.split(key, 4)

        env = camar_v0()
        assert env is not None

        obs, state = env.reset(key_r)
        assert obs is not None
        assert state is not None

        actions = env.action_spaces.sample(key_a)
        assert actions.shape == (env.num_agents, env.action_size)

        obs_next, state_next, reward, done, info = env.step(key_s, state, actions)
        assert obs_next is not None
        assert state_next is not None
        assert reward is not None
        assert done is not None
        assert isinstance(info, dict)

    def test_quickstart_vectorized_env(self):
        key = jax.random.key(0)
        key, key_r, key_a, key_s = jax.random.split(key, 4)

        num_envs = 10  # Reduced from 1000 for faster testing

        env = camar_v0()

        action_sampler = jax.jit(
            jax.vmap(
                env.action_spaces.sample,
                in_axes=[
                    0,
                ],
            )
        )
        env_reset_fn = jax.jit(
            jax.vmap(
                env.reset,
                in_axes=[
                    0,
                ],
            )
        )
        env_step_fn = jax.jit(
            jax.vmap(
                env.step,
                in_axes=[
                    0,
                    0,
                    0,
                ],
            )
        )

        keys_r_v = jax.random.split(key_r, num_envs)
        keys_a_v = jax.random.split(key_a, num_envs)
        keys_s_v = jax.random.split(key_s, num_envs)

        obs, state = env_reset_fn(keys_r_v)
        assert obs.shape == (num_envs, env.num_agents, env.observation_size)

        actions = action_sampler(keys_a_v)
        assert actions.shape == (num_envs, env.num_agents, env.action_size)

        obs_next, state_next, reward, done, info = env_step_fn(keys_s_v, state, actions)
        assert obs_next.shape == (num_envs, env.num_agents, env.observation_size)
        assert reward.shape == (num_envs, env.num_agents, 1)
        assert done.shape == (num_envs,)
        assert isinstance(info, dict)

    def test_wrappers_example(self):
        num_envs = 10  # Reduced from 1000 for faster testing
        env = OptimisticResetVecEnvWrapper(
            env=camar_v0(),
            num_envs=num_envs,
            reset_ratio=2,  # reduced from 20 for faster testing
        )
        assert env is not None

        key = jax.random.key(0)
        key_reset, key_step, key_action = jax.random.split(key, 3)

        obs, state = env.reset(key_reset)
        assert obs.shape[0] == num_envs

        key_actions = jax.random.split(key_action, num_envs)
        actions = jax.vmap(env.action_spaces.sample)(key_actions)
        assert actions.shape == (num_envs, env.num_agents, env.action_size)

        obs_next, state_next, reward, done, info = env.step(key_step, state, actions)
        assert obs_next.shape[0] == num_envs
        assert reward.shape[0] == num_envs
        assert done.shape[0] == num_envs

    def test_maps_example_creation(self):
        map_str_readme = """
        .....#.....
        .....#.....
        ...........
        .....#.....
        .....#.....
        #.####.....
        .....###.##
        .....#.....
        .....#.....
        ...........
        .....#.....
        """

        string_grid_map = string_grid(map_str=map_str_readme, num_agents=8)
        random_grid_map_custom = random_grid(num_agents=4, num_rows=10, num_cols=10)
        labmaze_map = labmaze_grid(
            num_maps=2, num_agents=3, height=7, width=7
        )  # Reduced for testing

        env1 = camar_v0(string_grid_map)
        env2 = camar_v0(random_grid_map_custom)
        env3 = camar_v0(labmaze_map)

        assert isinstance(env1.map_generator, string_grid)
        assert isinstance(env2.map_generator, random_grid)
        assert isinstance(env3.map_generator, labmaze_grid)
        assert env1.num_agents == 8
        assert env2.num_agents == 4
        assert env3.num_agents == 3

        env1_str = camar_v0("string_grid", map_str=map_str_readme, num_agents=8)
        env2_str = camar_v0("random_grid", num_agents=4, num_rows=10, num_cols=10)
        env3_str = camar_v0("labmaze_grid", num_maps=2, num_agents=3, height=7, width=7)

        assert isinstance(env1_str.map_generator, string_grid)
        assert isinstance(env2_str.map_generator, random_grid)
        assert isinstance(env3_str.map_generator, labmaze_grid)
        assert env1_str.num_agents == 8
        assert env2_str.num_agents == 4
        assert env3_str.num_agents == 3

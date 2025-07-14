import jax

from camar import camar_v0
from camar.maps import labmaze_grid, random_grid, string_grid
from camar.dynamics import DiffDriveDynamic, HolonomicDynamic, MixedDynamic
from camar.wrappers import OptimisticResetVecEnvWrapper


class TestReadmeExamples:
    def test_quickstart_single_env(self):
        key = jax.random.key(0)
        key, key_r, key_a, key_s = jax.random.split(key, 4)

        # Create environment (default: random_grid map with holonomic dynamics)
        env = camar_v0()
        assert env is not None

        reset_fn = jax.jit(env.reset)
        step_fn = jax.jit(env.step)

        # Reset the environment
        obs, state = reset_fn(key_r)
        assert obs is not None
        assert state is not None

        # Sample random actions
        actions = env.action_spaces.sample(key_a)
        assert actions.shape == (env.num_agents, env.action_size)

        # Step the environment
        obs_next, state_next, reward, done, info = step_fn(key_s, state, actions)
        assert obs_next is not None
        assert state_next is not None
        assert reward is not None
        assert done is not None
        assert isinstance(info, dict)

    def test_quickstart_vectorized_env(self):
        key = jax.random.key(0)
        key, key_r, key_a, key_s = jax.random.split(key, 4)

        # Setup for 1000 parallel environments (reduced for testing)
        num_envs = 10
        env = camar_v0()

        # Create vectorized functions
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

        # Generate keys for each environment
        keys_r_v = jax.random.split(key_r, num_envs)
        keys_a_v = jax.random.split(key_a, num_envs)
        keys_s_v = jax.random.split(key_s, num_envs)

        # Use as before
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
            reset_ratio=2,  # reduced from 200 for faster testing
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
        # Define a custom map layout for string_grid
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

        # Create maps
        string_grid_map = string_grid(map_str=map_str_readme, num_agents=8)
        random_grid_map_custom = random_grid(num_agents=4, num_rows=10, num_cols=10)

        # Use maps directly
        env1 = camar_v0(string_grid_map)
        env2 = camar_v0(random_grid_map_custom)

        assert isinstance(env1.map_generator, string_grid)
        assert isinstance(env2.map_generator, random_grid)
        assert env1.num_agents == 8
        assert env2.num_agents == 4

        # Or specify by name
        env1_str = camar_v0(
            "string_grid", map_kwargs={"map_str": map_str_readme, "num_agents": 8}
        )
        env2_str = camar_v0(
            "random_grid", map_kwargs={"num_agents": 4, "num_rows": 10, "num_cols": 10}
        )

        assert isinstance(env1_str.map_generator, string_grid)
        assert isinstance(env2_str.map_generator, random_grid)
        assert env1_str.num_agents == 8
        assert env2_str.num_agents == 4

        # labmaze is not supported by python=3.13
        try:
            labmaze_map = labmaze_grid(
                num_maps=2, num_agents=3, height=7, width=7
            )  # Reduced for testing
            env3 = camar_v0(labmaze_map)

            assert isinstance(env3.map_generator, labmaze_grid)
            assert env3.num_agents == 3

            env3_str = camar_v0(
                "labmaze_grid",
                map_kwargs={"num_maps": 2, "num_agents": 3, "height": 7, "width": 7},
            )

            assert isinstance(env3_str.map_generator, labmaze_grid)
            assert env3_str.num_agents == 3

        except ModuleNotFoundError:
            pass

    def test_dynamics_builtin_examples(self):
        from camar.dynamics import DiffDriveDynamic, HolonomicDynamic

        # Differential drive robots (like wheeled robots)
        diffdrive = DiffDriveDynamic(mass=1.0)

        # Holonomic robots (like omni-directional robots)
        holonomic = HolonomicDynamic(dt=0.001)

        # Use different dynamics
        env1 = camar_v0(dynamic=diffdrive)
        env2 = camar_v0(dynamic=holonomic)

        assert isinstance(env1.dynamic, DiffDriveDynamic)
        assert isinstance(env2.dynamic, HolonomicDynamic)
        assert env1.dynamic.mass == 1.0
        assert env2.dynamic.dt == 0.001

        # Or specify by name
        env1_str = camar_v0(dynamic="DiffDriveDynamic", dynamic_kwargs={"mass": 1.0})
        env2_str = camar_v0(dynamic="HolonomicDynamic", dynamic_kwargs={"dt": 0.001})

        assert isinstance(env1_str.dynamic, DiffDriveDynamic)
        assert isinstance(env2_str.dynamic, HolonomicDynamic)
        assert env1_str.dynamic.mass == 1.0
        assert env2_str.dynamic.dt == 0.001

    def test_dynamics_heterogeneous_example(self):
        # Define different dynamics for different agent groups
        dynamics_batch = [
            DiffDriveDynamic(mass=1.0),
            HolonomicDynamic(mass=10.0),
        ]
        num_agents_batch = [8, 24]  # 8 diffdrive + 24 holonomic = 32 total

        mixed_dynamic = MixedDynamic(
            dynamics_batch=dynamics_batch,
            num_agents_batch=num_agents_batch,
        )

        # Create environment with mixed dynamics
        env = camar_v0(
            map_generator="random_grid",
            dynamic=mixed_dynamic,
            map_kwargs={"num_agents": sum(num_agents_batch)},
        )

        assert isinstance(env.dynamic, MixedDynamic)
        assert env.num_agents == 32
        assert len(env.dynamic.dynamics_batch) == 2
        assert env.dynamic.num_agents_batch == [8, 24]

        # Or specify by name
        env_str = camar_v0(
            map_generator="random_grid",
            dynamic="MixedDynamic",
            map_kwargs={"num_agents": sum(num_agents_batch)},
            dynamic_kwargs={
                "dynamics_batch": dynamics_batch,
                "num_agents_batch": num_agents_batch
            },
        )

        assert isinstance(env_str.dynamic, MixedDynamic)
        assert env_str.num_agents == 32
        assert len(env_str.dynamic.dynamics_batch) == 2
        assert env_str.dynamic.num_agents_batch == [8, 24]

    def test_dynamics_custom_example(self):
        """Test the custom dynamics example from README"""
        from camar.dynamics import BaseDynamic, PhysicalState
        import jax.numpy as jnp
        from jax.typing import ArrayLike
        from flax import struct

        @struct.dataclass
        class CustomState(PhysicalState):
            agent_pos: ArrayLike  # mandatory field
            agent_vel: ArrayLike
            custom_field: ArrayLike  # Add your custom state fields

            @classmethod
            def create(cls, key, agent_pos):
                num_agents = agent_pos.shape[0]
                return cls(
                    agent_pos=agent_pos,
                    agent_vel=jnp.zeros((num_agents, 2)),
                    custom_field=jnp.zeros((num_agents, 1))
                )

        class CustomDynamic(BaseDynamic):
            def __init__(self, custom_param=1.0, dt=0.01):
                self.custom_param = custom_param
                self._dt = dt

            @property
            def action_size(self) -> int:
                return 2  # Your action space size

            @property
            def dt(self) -> float:
                return self._dt

            @property
            def state_class(self):
                return CustomState

            def integrate(self, key, force, physical_state, actions):
                # Your custom integration logic
                pos = physical_state.agent_pos
                vel = physical_state.agent_vel
                custom = physical_state.custom_field

                # Update state according to your dynamics
                new_vel = vel + (force + actions * self.custom_param) / 1.0 * self.dt
                new_pos = pos + new_vel * self.dt
                new_custom = custom + actions[:, 0:1] * self.dt

                return physical_state.replace(
                    agent_pos=new_pos,
                    agent_vel=new_vel,
                    custom_field=new_custom
                )

        # Test custom dynamics
        custom_dynamic = CustomDynamic(custom_param=2.0, dt=0.02)
        env = camar_v0(dynamic=custom_dynamic)

        assert isinstance(env.dynamic, CustomDynamic)
        assert env.dynamic.custom_param == 2.0
        assert env.dynamic.dt == 0.02
        assert env.dynamic.action_size == 2
        assert env.dynamic.state_class == CustomState

        # Test integration
        key = jax.random.key(0)
        key, key_r, key_a, key_s = jax.random.split(key, 4)

        obs, state = env.reset(key_r)
        actions = env.action_spaces.sample(key_a)
        obs_next, state_next, reward, done, info = env.step(key_s, state, actions)

        assert obs is not None
        assert state is not None
        assert obs_next is not None
        assert state_next is not None
        assert reward is not None
        assert done is not None

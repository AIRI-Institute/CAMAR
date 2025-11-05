import jax

from camar import camar_v0
from camar.dynamics import DiffDriveDynamic, HolonomicDynamic, MixedDynamic
from camar.maps import labmaze_grid, random_grid, string_grid
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
        env1_str = camar_v0("string_grid", map_kwargs={"map_str": map_str_readme, "num_agents": 8})
        env2_str = camar_v0("random_grid", map_kwargs={"num_agents": 4, "num_rows": 10, "num_cols": 10})

        assert isinstance(env1_str.map_generator, string_grid)
        assert isinstance(env2_str.map_generator, random_grid)
        assert env1_str.num_agents == 8
        assert env2_str.num_agents == 4

        # labmaze is not supported by python=3.13
        try:
            labmaze_map = labmaze_grid(num_maps=2, num_agents=3, height=7, width=7)  # Reduced for testing
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

        # Test full environment pipeline with jitted reset and step
        key = jax.random.key(0)
        key_r, key_a, key_s = jax.random.split(key, 3)

        reset_fn1 = jax.jit(env1.reset)
        step_fn1 = jax.jit(env1.step)
        obs1, state1 = reset_fn1(key_r)
        actions1 = env1.action_spaces.sample(key_a)
        obs_next1, state_next1, reward1, done1, info1 = step_fn1(key_s, state1, actions1)

        assert obs1.shape == (env1.num_agents, env1.observation_size)
        assert obs_next1.shape == (env1.num_agents, env1.observation_size)
        assert reward1.shape == (env1.num_agents, 1)
        assert isinstance(done1, jax.Array)

        reset_fn2 = jax.jit(env2.reset)
        step_fn2 = jax.jit(env2.step)
        obs2, state2 = reset_fn2(key_r)
        actions2 = env2.action_spaces.sample(key_a)
        obs_next2, state_next2, reward2, done2, info2 = step_fn2(key_s, state2, actions2)

        assert obs2.shape == (env2.num_agents, env2.observation_size)
        assert obs_next2.shape == (env2.num_agents, env2.observation_size)
        assert reward2.shape == (env2.num_agents, 1)
        assert isinstance(done2, jax.Array)

        # Or specify by name
        env1_str = camar_v0(dynamic="DiffDriveDynamic", dynamic_kwargs={"mass": 1.0})
        env2_str = camar_v0(dynamic="HolonomicDynamic", dynamic_kwargs={"dt": 0.001})

        assert isinstance(env1_str.dynamic, DiffDriveDynamic)
        assert isinstance(env2_str.dynamic, HolonomicDynamic)
        assert env1_str.dynamic.mass == 1.0
        assert env2_str.dynamic.dt == 0.001

        # Test string-based environments with jitted reset and step
        reset_fn1_str = jax.jit(env1_str.reset)
        step_fn1_str = jax.jit(env1_str.step)
        obs1_str, state1_str = reset_fn1_str(key_r)
        actions1_str = env1_str.action_spaces.sample(key_a)
        obs_next1_str, state_next1_str, reward1_str, done1_str, info1_str = step_fn1_str(
            key_s, state1_str, actions1_str
        )

        assert obs1_str.shape == (env1_str.num_agents, env1_str.observation_size)
        assert obs_next1_str.shape == (env1_str.num_agents, env1_str.observation_size)
        assert reward1_str.shape == (env1_str.num_agents, 1)

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
        key = jax.random.key(0)
        key_r, key_a, key_s = jax.random.split(key, 3)

        reset_fn = jax.jit(env.reset)
        step_fn = jax.jit(env.step)
        obs, state = reset_fn(key_r)
        actions = env.action_spaces.sample(key_a)
        obs_next, state_next, reward, done, info = step_fn(key_s, state, actions)

        assert obs is not None
        assert state is not None
        assert obs.shape == (env.num_agents, env.observation_size)
        assert obs_next.shape == (env.num_agents, env.observation_size)
        assert reward.shape == (env.num_agents, 1)

        assert isinstance(env.dynamic, MixedDynamic)
        assert env.num_agents == 32
        assert len(env.dynamic.dynamics_batch) == 2
        assert env.dynamic.num_agents_batch == [8, 24]

        # Or specify by name
        env_str = camar_v0(
            map_generator="random_grid",
            dynamic="MixedDynamic",
            map_kwargs={"num_agents": sum(num_agents_batch)},
            dynamic_kwargs={"dynamics_batch": dynamics_batch, "num_agents_batch": num_agents_batch},
        )

        assert isinstance(env_str.dynamic, MixedDynamic)
        assert env_str.num_agents == 32
        assert len(env_str.dynamic.dynamics_batch) == 2
        assert env_str.dynamic.num_agents_batch == [8, 24]

        # Test string-based mixed dynamic with jitted reset and step
        reset_fn_str = jax.jit(env_str.reset)
        step_fn_str = jax.jit(env_str.step)
        obs_str, state_str = reset_fn_str(key_r)
        actions_str = env_str.action_spaces.sample(key_a)
        obs_next_str, state_next_str, reward_str, done_str, info_str = step_fn_str(
            key_s, state_str, actions_str
        )

        assert obs_str.shape == (env_str.num_agents, env_str.observation_size)
        assert obs_next_str.shape == (env_str.num_agents, env_str.observation_size)
        assert reward_str.shape == (env_str.num_agents, 1)

    def test_maps_partial_registration_example(self):
        """Test the functools.partial map registration example from README."""
        from functools import partial

        from camar.maps import random_grid
        from camar.registry import register_map_class

        # Create and register a pre-configured map variant
        SmallDenseGrid = partial(
            random_grid,
            num_rows=10,
            num_cols=10,
            obstacle_density=0.30,
            num_agents=8,
        )
        register_map_class("SmallDenseGrid", SmallDenseGrid)

        # Usable by string
        env1 = camar_v0(map_generator="SmallDenseGrid")
        assert env1.num_agents == 8

        # Test with jitted reset and step
        key = jax.random.key(0)
        key_r, key_a, key_s = jax.random.split(key, 3)
        reset_fn1 = jax.jit(env1.reset)
        step_fn1 = jax.jit(env1.step)
        obs1, state1 = reset_fn1(key_r)
        actions1 = env1.action_spaces.sample(key_a)
        obs_next1, state_next1, reward1, done1, info1 = step_fn1(key_s, state1, actions1)

        assert obs1.shape == (env1.num_agents, env1.observation_size)
        assert obs_next1.shape == (env1.num_agents, env1.observation_size)
        assert reward1.shape == (env1.num_agents, 1)

        # Usable by callable
        env2 = camar_v0(map_generator=SmallDenseGrid())
        assert env2.num_agents == 8

        # Test with jitted reset and step
        reset_fn2 = jax.jit(env2.reset)
        step_fn2 = jax.jit(env2.step)
        obs2, state2 = reset_fn2(key_r)
        actions2 = env2.action_spaces.sample(key_a)
        obs_next2, state_next2, reward2, done2, info2 = step_fn2(key_s, state2, actions2)

        assert obs2.shape == (env2.num_agents, env2.observation_size)
        assert obs_next2.shape == (env2.num_agents, env2.observation_size)

        # Usable by instance with override
        env3 = camar_v0(map_generator=SmallDenseGrid(num_agents=12))
        assert env3.num_agents == 12

        # Test with jitted reset and step
        reset_fn3 = jax.jit(env3.reset)
        step_fn3 = jax.jit(env3.step)
        obs3, state3 = reset_fn3(key_r)
        actions3 = env3.action_spaces.sample(key_a)
        obs_next3, state_next3, reward3, done3, info3 = step_fn3(key_s, state3, actions3)

        assert obs3.shape == (env3.num_agents, env3.observation_size)
        assert obs_next3.shape == (env3.num_agents, env3.observation_size)

    def test_dynamics_partial_registration_example(self):
        """Test the functools.partial dynamics registration example from README."""
        from functools import partial

        from camar.dynamics import HolonomicDynamic
        from camar.registry import register_dynamic_class

        SlowHolonomic = partial(HolonomicDynamic, max_speed=1.0, accel=4.0)
        register_dynamic_class("SlowHolonomic", SlowHolonomic)

        # By string
        env1 = camar_v0(dynamic="SlowHolonomic")
        assert isinstance(env1.dynamic, HolonomicDynamic)
        assert env1.dynamic.max_speed == 1.0

        # Test with jitted reset and step
        key = jax.random.key(0)
        key_r, key_a, key_s = jax.random.split(key, 3)
        reset_fn1 = jax.jit(env1.reset)
        step_fn1 = jax.jit(env1.step)
        obs1, state1 = reset_fn1(key_r)
        actions1 = env1.action_spaces.sample(key_a)
        obs_next1, state_next1, reward1, done1, info1 = step_fn1(key_s, state1, actions1)

        assert obs1.shape == (env1.num_agents, env1.observation_size)
        assert obs_next1.shape == (env1.num_agents, env1.observation_size)
        assert reward1.shape == (env1.num_agents, 1)

        # By callable
        env2 = camar_v0(dynamic=SlowHolonomic())
        assert isinstance(env2.dynamic, HolonomicDynamic)
        assert env2.dynamic.max_speed == 1.0

        # Test with jitted reset and step
        reset_fn2 = jax.jit(env2.reset)
        step_fn2 = jax.jit(env2.step)
        obs2, state2 = reset_fn2(key_r)
        actions2 = env2.action_spaces.sample(key_a)
        obs_next2, state_next2, reward2, done2, info2 = step_fn2(key_s, state2, actions2)

        assert obs2.shape == (env2.num_agents, env2.observation_size)
        assert obs_next2.shape == (env2.num_agents, env2.observation_size)

        # By instance with override
        env3 = camar_v0(dynamic=SlowHolonomic(dt=0.02))
        assert isinstance(env3.dynamic, HolonomicDynamic)
        assert env3.dynamic.dt == 0.02

        # Test with jitted reset and step
        reset_fn3 = jax.jit(env3.reset)
        step_fn3 = jax.jit(env3.step)
        obs3, state3 = reset_fn3(key_r)
        actions3 = env3.action_spaces.sample(key_a)
        obs_next3, state_next3, reward3, done3, info3 = step_fn3(key_s, state3, actions3)

        assert obs3.shape == (env3.num_agents, env3.observation_size)
        assert obs_next3.shape == (env3.num_agents, env3.observation_size)

    def test_maps_decorator_registration_example(self):
        """Test the maps decorator registration example from README."""
        import jax.numpy as jnp

        from camar.maps.base import base_map
        from camar.registry import register_map

        @register_map("MyMap")
        class MyMap(base_map):
            def __init__(self, height: float = 3.0, num_agents: int = 2):
                self._height = height
                self._num_agents = num_agents
                self._num_landmarks = 1
                super().__init__()

            def setup_rad(self):
                self.agent_rad = 0.1
                self.landmark_rad = 0.1
                self.goal_rad = 0.1
                self.proportional_goal_rad = False

            @property
            def height(self):
                return self._height

            @property
            def width(self):
                return self._height

            @property
            def num_agents(self) -> int:
                return self._num_agents

            @property
            def num_landmarks(self) -> int:
                return self._num_landmarks

            def reset(self, key):
                sizes = self.generate_sizes(key)
                return (
                    key,
                    jnp.zeros((1, 2)),
                    jnp.zeros((self._num_agents, 2)),
                    jnp.zeros((self._num_agents, 2)),
                    sizes,
                )

        env = camar_v0(
            map_generator="MyMap",
            map_kwargs={
                "height": 7.0,
                "num_agents": 3,
            },
        )
        assert type(env.map_generator).__name__ == "MyMap"
        assert env.height == 7.0
        assert env.num_agents == 3

        # Test with jitted reset and step
        key = jax.random.key(0)
        key_r, key_a, key_s = jax.random.split(key, 3)
        reset_fn = jax.jit(env.reset)
        step_fn = jax.jit(env.step)
        obs, state = reset_fn(key_r)
        actions = env.action_spaces.sample(key_a)
        obs_next, state_next, reward, done, info = step_fn(key_s, state, actions)

        assert obs.shape == (env.num_agents, env.observation_size)
        assert obs_next.shape == (env.num_agents, env.observation_size)
        assert reward.shape == (env.num_agents, 1)
        assert state.physical_state.agent_pos.shape == (env.num_agents, 2)

    def test_maps_function_registration_example(self):
        """Test the maps function registration example from README."""
        import jax.numpy as jnp

        from camar.maps.base import base_map
        from camar.registry import register_map_class

        class MyMap2(base_map):
            def __init__(self, height: float = 3.0, num_agents: int = 2):
                self._height = height
                self._num_agents = num_agents
                self._num_landmarks = 1
                super().__init__()

            def setup_rad(self):
                self.agent_rad = 0.1
                self.landmark_rad = 0.1
                self.goal_rad = 0.1
                self.proportional_goal_rad = False

            @property
            def height(self):
                return self._height

            @property
            def width(self):
                return self._height

            @property
            def num_agents(self) -> int:
                return self._num_agents

            @property
            def num_landmarks(self) -> int:
                return self._num_landmarks

            def reset(self, key):
                sizes = self.generate_sizes(key)
                return (
                    key,
                    jnp.zeros((1, 2)),
                    jnp.zeros((self._num_agents, 2)),
                    jnp.zeros((self._num_agents, 2)),
                    sizes,
                )

        register_map_class("OtherMap", MyMap2)
        env = camar_v0(
            map_generator="OtherMap",
            map_kwargs={
                "height": 7.0,
                "num_agents": 4,
            },
        )
        assert type(env.map_generator).__name__ == "MyMap2"
        assert env.height == 7.0
        assert env.num_agents == 4

        # Test with jitted reset and step
        key = jax.random.key(0)
        key_r, key_a, key_s = jax.random.split(key, 3)
        reset_fn = jax.jit(env.reset)
        step_fn = jax.jit(env.step)
        obs, state = reset_fn(key_r)
        actions = env.action_spaces.sample(key_a)
        obs_next, state_next, reward, done, info = step_fn(key_s, state, actions)

        assert obs.shape == (env.num_agents, env.observation_size)
        assert obs_next.shape == (env.num_agents, env.observation_size)
        assert reward.shape == (env.num_agents, 1)
        assert state.physical_state.agent_pos.shape == (env.num_agents, 2)

    def test_dynamics_decorator_registration_example(self):
        """Test the dynamics decorator registration example from README (updated)."""
        import jax.numpy as jnp
        from flax import struct
        from jax.typing import ArrayLike

        from camar.dynamics import BaseDynamic, PhysicalState
        from camar.registry import register_dynamic

        @struct.dataclass
        class CustomState(PhysicalState):
            agent_pos: ArrayLike
            agent_vel: ArrayLike
            count: ArrayLike

            @classmethod
            def create(cls, key, landmark_pos, agent_pos, goal_pos, sizes):
                n = agent_pos.shape[0]
                return cls(
                    agent_pos=agent_pos,
                    agent_vel=jnp.zeros((n, 2)),
                    count=jnp.zeros((n, 2)),
                )

        @register_dynamic("CustomDynamic")
        class CustomDynamic(BaseDynamic):
            def __init__(self, custom_param=1.0, dt=0.01, vel_counter_thr=0.01):
                self.custom_param = custom_param
                self._dt = dt
                self.vel_counter_thr = vel_counter_thr

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
                new_vel = vel + (force + actions * self.custom_param) * self.dt
                new_pos = pos + new_vel * self.dt

                # update counter
                new_count = jnp.where(
                    jnp.linalg.norm(new_vel, axis=-1, keepdims=True) > self.vel_counter_thr,
                    physical_state.count + 1,
                    physical_state.count,
                )
                new_physical_state = physical_state.replace(
                    agent_pos=new_pos,
                    agent_vel=new_vel,
                    count=new_count,
                )
                return new_physical_state

        env = camar_v0(
            dynamic="CustomDynamic",
            dynamic_kwargs={"custom_param": 2.0},
        )
        assert type(env.dynamic).__name__ == "CustomDynamic"
        assert env.dynamic.dt == 0.01

        # Test with jitted reset and step
        key = jax.random.key(0)
        key_r, key_a, key_s = jax.random.split(key, 3)
        reset_fn = jax.jit(env.reset)
        step_fn = jax.jit(env.step)
        obs, state = reset_fn(key_r)
        actions = env.action_spaces.sample(key_a)
        obs_next, state_next, reward, done, info = step_fn(key_s, state, actions)

        assert obs.shape == (env.num_agents, env.observation_size)
        assert obs_next.shape == (env.num_agents, env.observation_size)
        assert reward.shape == (env.num_agents, 1)
        assert state.physical_state.agent_pos.shape == (env.num_agents, 2)
        assert hasattr(state.physical_state, "count")

    def test_dynamics_function_registration_example(self):
        """Test the dynamics function registration example from README."""
        import jax.numpy as jnp
        from flax import struct
        from jax.typing import ArrayLike

        from camar.dynamics import BaseDynamic, PhysicalState
        from camar.registry import register_dynamic_class

        # simple example
        @struct.dataclass
        class CustomState2(PhysicalState):
            agent_pos: ArrayLike
            agent_vel: ArrayLike

            @classmethod
            def create(cls, key, landmark_pos, agent_pos, goal_pos, sizes):
                n = agent_pos.shape[0]
                return cls(agent_pos=agent_pos, agent_vel=jnp.zeros((n, 2)))

        class CustomDynamic2(BaseDynamic):
            def __init__(self, dt=0.01):
                self._dt = dt

            @property
            def action_size(self) -> int:
                return 2

            @property
            def dt(self) -> float:
                return self._dt

            @property
            def state_class(self):
                return CustomState2

            def integrate(self, key, force, physical_state, actions):
                return physical_state

        register_dynamic_class("OtherDyn", CustomDynamic2)
        env = camar_v0(dynamic="OtherDyn")
        assert type(env.dynamic).__name__ == "CustomDynamic2"

        # Test with jitted reset and step
        key = jax.random.key(0)
        key_r, key_a, key_s = jax.random.split(key, 3)
        reset_fn = jax.jit(env.reset)
        step_fn = jax.jit(env.step)
        obs, state = reset_fn(key_r)
        actions = env.action_spaces.sample(key_a)
        obs_next, state_next, reward, done, info = step_fn(key_s, state, actions)

        assert obs.shape == (env.num_agents, env.observation_size)
        assert obs_next.shape == (env.num_agents, env.observation_size)
        assert reward.shape == (env.num_agents, 1)
        assert state.physical_state.agent_pos.shape == (env.num_agents, 2)

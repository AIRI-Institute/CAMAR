import jax
import jax.numpy as jnp
import pytest
from dummy_map import dummy_map

from camar import camar_v0
from camar.environment import Camar
from camar.maps import string_grid
from camar.dynamics import HolonomicDynamic, DiffDriveDynamic, MixedDynamic


@pytest.fixture
def key():
    return jax.random.key(0)


@pytest.fixture
def landmark_pos():
    return jnp.array([[0.0, 0.0]])


@pytest.fixture
def agent_pos():
    return jnp.array([[1.0, 1.0]])


@pytest.fixture
def goal_pos():
    return jnp.array([[2.0, 2.0]])


@pytest.fixture
def dummy_map_generator(landmark_pos, agent_pos, goal_pos):
    return dummy_map(landmark_pos=landmark_pos, agent_pos=agent_pos, goal_pos=goal_pos)


@pytest.fixture
def holonomic_dynamic():
    return HolonomicDynamic(accel=5.0, max_speed=6.0, damping=0.25, mass=1.0, dt=0.01)


@pytest.fixture
def diffdrive_dynamic():
    return DiffDriveDynamic(
        linear_speed_max=1.0,
        linear_speed_min=-1.0,
        angular_speed_max=2.0,
        angular_speed_min=-2.0,
        mass=1.0,
        dt=0.01,
    )


@pytest.fixture
def mixed_dynamic():
    holonomic = HolonomicDynamic(mass=1.0)
    diffdrive = DiffDriveDynamic(mass=2.0)
    return MixedDynamic(dynamics_batch=[holonomic, diffdrive], num_agents_batch=[2, 3])


def test_camar_v0_with_map_generator(key, dummy_map_generator):
    env = camar_v0(map_generator=dummy_map_generator)
    obs, state = env.reset(key)

    assert type(env) is Camar
    assert type(env.map_generator) is dummy_map
    assert env.height == 10.0
    assert env.width == 10.0
    assert env.num_landmarks == 1
    assert env.num_agents == 1
    assert jnp.allclose(state.landmark_pos, jnp.array([[0.0, 0.0]]))
    assert jnp.allclose(state.physical_state.agent_pos, jnp.array([[1.0, 1.0]]))
    assert jnp.allclose(state.goal_pos, jnp.array([[2.0, 2.0]]))


def test_camar_v0_with_string_grid_string():
    env = camar_v0(
        map_generator="string_grid",
        map_kwargs={"map_str": "...", "num_agents": 1},
    )

    assert type(env) is Camar
    assert type(env.map_generator) is string_grid
    assert jnp.isclose(env.height, 0.3)
    assert jnp.isclose(env.width, 0.5)
    assert env.num_landmarks == 12  # only borders
    assert env.num_agents == 1


def test_camar_v0_with_string_grid_class():
    map_generator = string_grid(map_str="...", num_agents=1)
    env = camar_v0(map_generator=map_generator)

    assert type(env) is Camar
    assert type(env.map_generator) is string_grid
    assert jnp.isclose(env.height, 0.3)
    assert jnp.isclose(env.width, 0.5)
    assert env.num_landmarks == 12  # only borders
    assert env.num_agents == 1


def test_camar_v0_with_holonomic_dynamic_string():
    env = camar_v0(
        dynamic="HolonomicDynamic",
        dynamic_kwargs={"accel": 10.0, "max_speed": 8.0, "damping": 0.5, "mass": 2.0, "dt": 0.02},
    )

    assert type(env) is Camar
    assert type(env.dynamic) is HolonomicDynamic
    assert env.dynamic.accel == 10.0
    assert env.dynamic.max_speed == 8.0
    assert env.dynamic.damping == 0.5
    assert env.dynamic.mass == 2.0
    assert env.dynamic.dt == 0.02


def test_camar_v0_with_holonomic_dynamic_class(holonomic_dynamic):
    env = camar_v0(dynamic=holonomic_dynamic)

    assert type(env) is Camar
    assert type(env.dynamic) is HolonomicDynamic
    assert env.dynamic.accel == 5.0
    assert env.dynamic.max_speed == 6.0
    assert env.dynamic.damping == 0.25
    assert env.dynamic.mass == 1.0
    assert env.dynamic.dt == 0.01


def test_camar_v0_with_diffdrive_dynamic_string():
    env = camar_v0(
        dynamic="DiffDriveDynamic",
        dynamic_kwargs={
            "linear_speed_max": 2.0,
            "linear_speed_min": -0.5,
            "angular_speed_max": 3.0,
            "angular_speed_min": -1.0,
            "mass": 2.0,
            "dt": 0.02,
        },
    )

    assert type(env) is Camar
    assert type(env.dynamic) is DiffDriveDynamic
    assert env.dynamic.linear_speed_max == 2.0
    assert env.dynamic.linear_speed_min == -0.5
    assert env.dynamic.angular_speed_max == 3.0
    assert env.dynamic.angular_speed_min == -1.0
    assert env.dynamic.mass == 2.0
    assert env.dynamic.dt == 0.02


def test_camar_v0_with_diffdrive_dynamic_class(diffdrive_dynamic):
    env = camar_v0(dynamic=diffdrive_dynamic)

    assert type(env) is Camar
    assert type(env.dynamic) is DiffDriveDynamic
    assert env.dynamic.linear_speed_max == 1.0
    assert env.dynamic.linear_speed_min == -1.0
    assert env.dynamic.angular_speed_max == 2.0
    assert env.dynamic.angular_speed_min == -2.0
    assert env.dynamic.mass == 1.0
    assert env.dynamic.dt == 0.01


def test_camar_v0_with_mixed_dynamic_class(mixed_dynamic):
    map_generator = string_grid(map_str=".....", num_agents=5)  # num_agents matched with mixed_dynamic
    env = camar_v0(map_generator=map_generator, dynamic=mixed_dynamic)

    assert type(env) is Camar
    assert type(env.dynamic) is MixedDynamic
    assert len(env.dynamic.dynamics_batch) == 2
    assert env.dynamic.num_agents_batch == [2, 3]
    assert env.dynamic.num_agents == 5
    assert env.num_agents == 5


def test_camar_v0_with_custom_map_and_dynamic(key, dummy_map_generator, holonomic_dynamic):
    env = camar_v0(map_generator=dummy_map_generator, dynamic=holonomic_dynamic)
    obs, state = env.reset(key)

    assert type(env) is Camar
    assert type(env.map_generator) is dummy_map
    assert type(env.dynamic) is HolonomicDynamic
    assert env.num_agents == 1
    assert env.dynamic.accel == 5.0
    assert env.dynamic.max_speed == 6.0


def test_camar_v0_with_string_grid_and_diffdrive(key, diffdrive_dynamic):
    env = camar_v0(
        map_generator="string_grid", map_kwargs={"map_str": "...", "num_agents": 2}, dynamic=diffdrive_dynamic
    )
    obs, state = env.reset(key)

    assert type(env) is Camar
    assert type(env.map_generator) is string_grid
    assert type(env.dynamic) is DiffDriveDynamic
    assert env.num_agents == 2
    assert env.dynamic.linear_speed_max == 1.0
    assert env.dynamic.angular_speed_max == 2.0


def test_camar_v0_with_mixed_dynamic_and_multiple_agents(key, mixed_dynamic):
    # Create a map generator with more free cells to support 5 agents
    map_str = """....."""
    map_generator = string_grid(map_str=map_str, num_agents=5)
    env = camar_v0(map_generator=map_generator, dynamic=mixed_dynamic)
    obs, state = env.reset(key)

    assert type(env) is Camar
    assert type(env.dynamic) is MixedDynamic
    assert env.num_agents == 5
    assert len(env.dynamic.dynamics_batch) == 2
    assert env.dynamic.num_agents_batch == [2, 3]

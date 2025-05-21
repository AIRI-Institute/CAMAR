import jax
import jax.numpy as jnp
import pytest

from camar import camar_v0
from camar.environment import Camar
from camar.maps import string_grid
from dummy_map import dummy_map


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
    assert jnp.allclose(state.agent_pos, jnp.array([[1.0, 1.0]]))
    assert jnp.allclose(state.goal_pos, jnp.array([[2.0, 2.0]]))


def test_camar_v0_with_string_grid_string():
    env = camar_v0(
        map_generator="string_grid",
        map_str="...",
        num_agents=1,
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

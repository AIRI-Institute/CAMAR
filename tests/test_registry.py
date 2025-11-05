import jax
import jax.numpy as jnp
import pytest

from camar.make_env import make_env as camar_v0
from camar.registry import (
    MAP_REGISTRY,
    DYNAMIC_REGISTRY,
    register_map,
    register_dynamic,
    register_map_class,
    register_dynamic_class,
)
from camar.maps.base import base_map
from camar.dynamics.base import BaseDynamic


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


def test_builtins_are_registered_via_decorators_import_side_effect():
    # Built-ins should be available by name after package import
    assert "random_grid" in MAP_REGISTRY
    assert "string_grid" in MAP_REGISTRY
    assert "HolonomicDynamic" in DYNAMIC_REGISTRY
    assert "DiffDriveDynamic" in DYNAMIC_REGISTRY


def test_register_map_decorator_and_use_string():
    @register_map("MyMap")
    class MyMap(base_map):
        def __init__(self, height: float = 3.14, num_agents: int = 2):
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

    env = camar_v0(map_generator="MyMap", map_kwargs={"height": 7.0, "num_agents": 3})
    assert type(env.map_generator).__name__ == "MyMap"
    assert env.height == 7.0
    assert env.num_agents == 3


def test_register_dynamic_decorator_and_use_string():
    @register_dynamic("MyDyn")
    class MyDyn(BaseDynamic):
        def __init__(self, dt: float = 0.03):
            self._dt = dt

        @property
        def action_size(self) -> int:
            return 2

        @property
        def dt(self) -> float:
            return self._dt

        @property
        def state_class(self):
            # Return any PhysicalState-like; tests won’t call integrate
            from camar.dynamics.holonomic import HolonomicState

            return HolonomicState

        def integrate(self, key, force, physical_state, actions):
            return physical_state

    env = camar_v0(dynamic="MyDyn", dynamic_kwargs={"dt": 0.07})
    assert type(env.dynamic).__name__ == "MyDyn"
    assert env.dynamic.dt == 0.07


def test_register_map_class_and_use_string(landmark_pos, agent_pos, goal_pos):
    from dummy_map import dummy_map

    # Programmatic registration of an existing class
    register_map_class("OtherMap", dummy_map)
    env = camar_v0(
        map_generator="OtherMap",
        map_kwargs={
            "height": 12.3,
            "num_agents": 2,
            "landmark_pos": landmark_pos,
            "agent_pos": agent_pos,
            "goal_pos": goal_pos,
        },
    )
    assert type(env.map_generator).__name__ == "dummy_map"
    assert env.height == 12.3
    assert env.num_agents == 2


def test_register_dynamic_class_and_use_string():
    class SimpleDyn(BaseDynamic):
        def __init__(self, dt: float = 0.05):
            self._dt = dt

        @property
        def action_size(self) -> int:
            return 2

        @property
        def dt(self) -> float:
            return self._dt

        @property
        def state_class(self):
            from camar.dynamics.holonomic import HolonomicState

            return HolonomicState

        def integrate(self, key, force, physical_state, actions):
            return physical_state

    register_dynamic_class("OtherDyn", SimpleDyn)
    env = camar_v0(dynamic="OtherDyn", dynamic_kwargs={"dt": 0.11})
    assert type(env.dynamic).__name__ == "SimpleDyn"
    assert env.dynamic.dt == 0.11


def test_class_and_instance_inputs_with_kwargs():
    # Class input for map
    class TinyMap(base_map):
        def __init__(self, h: float = 1.0):
            self._h = h
            self._num_agents = 1
            self._num_landmarks = 1
            super().__init__()

        def setup_rad(self):
            self.agent_rad = 0.1
            self.landmark_rad = 0.1
            self.goal_rad = 0.1
            self.proportional_goal_rad = False

        @property
        def height(self):
            return self._h

        @property
        def width(self):
            return self._h

        @property
        def num_agents(self) -> int:
            return self._num_agents

        @property
        def num_landmarks(self) -> int:
            return self._num_landmarks

        def reset(self, key):
            import jax.numpy as jnp

            sizes = self.generate_sizes(key)
            return key, jnp.zeros((1, 2)), jnp.zeros((1, 2)), jnp.zeros((1, 2)), sizes

    env = camar_v0(map_generator=TinyMap, map_kwargs={"h": 4.2})
    assert type(env.map_generator).__name__ == "TinyMap"
    assert env.height == 4.2

    # Instance input for dynamic
    class TinyDyn(BaseDynamic):
        def __init__(self, dt: float = 0.02):
            self._dt = dt

        @property
        def action_size(self) -> int:
            return 2

        @property
        def dt(self) -> float:
            return self._dt

        @property
        def state_class(self):
            from camar.dynamics.holonomic import HolonomicState

            return HolonomicState

        def integrate(self, key, force, physical_state, actions):
            return physical_state

    env = camar_v0(dynamic=TinyDyn(dt=0.09))
    assert type(env.dynamic).__name__ == "TinyDyn"
    assert env.dynamic.dt == 0.09


def test_unregistered_string_raises():
    with pytest.raises(TypeError):
        _ = camar_v0(map_generator="ThisDoesNotExist")

import jax
import jax.numpy as jnp
import pytest

from camar.dynamics import HolonomicDynamic, HolonomicState


@pytest.fixture
def key():
    return jax.random.key(0)


@pytest.fixture
def agent_pos():
    return jnp.array([[1.0, 1.0]])


@pytest.fixture
def landmark_pos():
    return jnp.array([[0.0, 0.0]])


@pytest.fixture
def goal_pos():
    return jnp.array([[2.0, 2.0]])


class TestHolonomicState:
    def test_holonomic_state_creation(self, key, agent_pos, landmark_pos, goal_pos):
        state = HolonomicState.create(key, landmark_pos, agent_pos, goal_pos, sizes=None)

        assert state.agent_pos.shape == (1, 2)
        assert state.agent_vel.shape == (1, 2)
        assert jnp.array_equal(state.agent_pos, agent_pos)
        assert jnp.allclose(state.agent_vel, jnp.zeros((1, 2)))

    def test_holonomic_state_agent_vel_initialization(self, key, agent_pos, landmark_pos, goal_pos):
        state = HolonomicState.create(key, landmark_pos, agent_pos, goal_pos, sizes=None)

        assert state.agent_vel.shape == (1, 2)
        assert jnp.allclose(state.agent_vel, jnp.zeros((1, 2)))


class TestHolonomicDynamic:
    def test_holonomic_dynamic_creation_default(self):
        dynamic = HolonomicDynamic()

        assert dynamic.accel == 5.0
        assert dynamic.max_speed == 6.0
        assert dynamic.damping == 0.25
        assert dynamic.mass == 1.0
        assert dynamic.dt == 0.01

    def test_holonomic_dynamic_creation_custom(self):
        dynamic = HolonomicDynamic(accel=10.0, max_speed=8.0, damping=0.5, mass=2.0, dt=0.02)

        assert dynamic.accel == 10.0
        assert dynamic.max_speed == 8.0
        assert dynamic.damping == 0.5
        assert dynamic.mass == 2.0
        assert dynamic.dt == 0.02

    def test_holonomic_dynamic_properties(self):
        dynamic = HolonomicDynamic()

        assert dynamic.action_size == 2
        assert dynamic.dt == 0.01
        assert dynamic.state_class == HolonomicState

    def test_holonomic_dynamic_integration_basic(self, key, agent_pos):
        dynamic = HolonomicDynamic()

        state = HolonomicState.create(key, jnp.zeros((1, 2)), agent_pos, agent_pos + 1.0, sizes=None)

        # No forces, no actions
        force = jnp.zeros((1, 2))
        actions = jnp.zeros((1, 2))

        new_state = dynamic.integrate(key, force, state, actions)

        # Position should remain the same (no velocity)
        assert jnp.allclose(new_state.agent_pos, state.agent_pos)
        # Velocity should be damped
        assert jnp.allclose(new_state.agent_vel, state.agent_vel * (1 - dynamic.damping))

    def test_holonomic_dynamic_jit_compatibility(self, key, agent_pos):
        dynamic = HolonomicDynamic()

        state = HolonomicState.create(key, jnp.zeros((1, 2)), agent_pos, agent_pos + 1.0, sizes=None)

        force = jnp.array([[1.0, 0.0]])
        actions = jnp.array([[0.5, 0.0]])

        # JIT the integrate function
        jitted_integrate = jax.jit(dynamic.integrate)
        new_state = jitted_integrate(key, force, state, actions)

        assert new_state.agent_pos.shape == (1, 2)
        assert new_state.agent_vel.shape == (1, 2)

import jax
import jax.numpy as jnp
import pytest

from camar.dynamics import DiffDriveDynamic, DiffDriveState


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


class TestDiffDriveState:
    def test_diffdrive_state_creation(self, key, agent_pos, landmark_pos, goal_pos):
        state = DiffDriveState.create(key, landmark_pos, agent_pos, goal_pos, sizes=None)

        assert state.agent_pos.shape == (1, 2)
        assert state.agent_vel.shape == (1, 2)
        assert state.agent_angle.shape == (1, 1)
        assert jnp.array_equal(state.agent_pos, agent_pos)
        assert jnp.allclose(state.agent_vel, jnp.zeros((1, 2)))
        # angle should point towards goal
        expected_angle = jnp.arctan2(
            goal_pos[:, 1] - agent_pos[:, 1], goal_pos[:, 0] - agent_pos[:, 0]
        ).reshape(1, 1)
        assert jnp.allclose(state.agent_angle, expected_angle)

    def test_diffdrive_state_agent_angle_initialization(self, key, agent_pos, landmark_pos, goal_pos):
        state = DiffDriveState.create(key, landmark_pos, agent_pos, goal_pos, sizes=None)

        assert state.agent_angle.shape == (1, 1)
        expected_angle = jnp.arctan2(
            goal_pos[:, 1] - agent_pos[:, 1], goal_pos[:, 0] - agent_pos[:, 0]
        ).reshape(1, 1)
        assert jnp.allclose(state.agent_angle, expected_angle)


class TestDiffDriveDynamic:
    def test_diffdrive_dynamic_creation_default(self):
        dynamic = DiffDriveDynamic()

        assert dynamic.linear_speed_max == 1.0
        assert dynamic.linear_speed_min == -1.0
        assert dynamic.angular_speed_max == 2.0
        assert dynamic.angular_speed_min == -2.0
        assert dynamic.mass == 1.0
        assert dynamic.dt == 0.01

    def test_diffdrive_dynamic_creation_custom(self):
        dynamic = DiffDriveDynamic(
            linear_speed_max=2.0,
            linear_speed_min=-0.5,
            angular_speed_max=3.0,
            angular_speed_min=-1.0,
            mass=2.0,
            dt=0.02,
        )

        assert dynamic.linear_speed_max == 2.0
        assert dynamic.linear_speed_min == -0.5
        assert dynamic.angular_speed_max == 3.0
        assert dynamic.angular_speed_min == -1.0
        assert dynamic.mass == 2.0
        assert dynamic.dt == 0.02

    def test_diffdrive_dynamic_properties(self):
        dynamic = DiffDriveDynamic()

        assert dynamic.action_size == 2
        assert dynamic.dt == 0.01
        assert dynamic.state_class == DiffDriveState

    def test_diffdrive_dynamic_integration_basic(self, key, agent_pos):
        dynamic = DiffDriveDynamic()

        # initialize facing goal (angle is irrelevant for zero actions)
        state = DiffDriveState.create(key, jnp.zeros((1, 2)), agent_pos, agent_pos + 1.0, sizes=None)

        force = jnp.zeros((1, 2))
        actions = jnp.zeros((1, 2))

        new_state = dynamic.integrate(key, force, state, actions)

        assert jnp.allclose(new_state.agent_pos, state.agent_pos)
        assert jnp.allclose(new_state.agent_angle, state.agent_angle)

    def test_diffdrive_dynamic_integration_forward_motion(self, key, agent_pos):
        dynamic = DiffDriveDynamic()
        state = DiffDriveState(
            agent_pos=agent_pos,
            agent_vel=jnp.array([[0.0, 0.0]]),
            agent_angle=jnp.array([[0.0]]),
        )  # not using create to ensure angle=0.0

        # Forward motion action
        force = jnp.array([[0.0, 0.0]])
        actions = jnp.array([[1.0, 0.0]])  # Full forward, no turning

        new_state = dynamic.integrate(key, force, state, actions)

        # Should move forward in x direction
        assert new_state.agent_pos[0, 0] > state.agent_pos[0, 0]  # x increased
        assert jnp.allclose(new_state.agent_pos[0, 1], state.agent_pos[0, 1])  # y unchanged
        assert jnp.allclose(new_state.agent_angle, state.agent_angle)  # angle unchanged

    def test_diffdrive_dynamic_angle_normalization(self, key, agent_pos):
        dynamic = DiffDriveDynamic()
        state = DiffDriveState(
            agent_pos=agent_pos,
            agent_vel=jnp.zeros((1, 2)),
            agent_angle=jnp.array([[3.0 * jnp.pi]]),
        )

        force = jnp.zeros((1, 2))
        actions = jnp.zeros((1, 2))

        new_state = dynamic.integrate(key, force, state, actions)

        # Angle should be normalized to [-pi, pi]
        assert -jnp.pi <= new_state.agent_angle[0, 0] <= jnp.pi

    def test_diffdrive_dynamic_negative_angle_normalization(self, key, agent_pos):
        dynamic = DiffDriveDynamic()
        state = DiffDriveState(
            agent_pos=agent_pos,
            agent_vel=jnp.zeros((1, 2)),
            agent_angle=jnp.array([[-2.0 * jnp.pi]]),
        )

        force = jnp.zeros((1, 2))
        actions = jnp.zeros((1, 2))

        new_state = dynamic.integrate(key, force, state, actions)

        # Angle should be normalized to [-pi, pi]
        assert -jnp.pi <= new_state.agent_angle[0, 0] <= jnp.pi

    def test_diffdrive_dynamic_jit_compatibility(self, key, agent_pos):
        dynamic = DiffDriveDynamic()

        # two-agent state for batching
        agent_pos_2 = jnp.array([[1.0, 1.0], [0.0, 0.0]])
        goal_pos_2 = jnp.array([[2.0, 1.0], [0.0, 1.0]])
        landmark_pos_2 = jnp.zeros((1, 2))
        state = DiffDriveState.create(key, landmark_pos_2, agent_pos_2, goal_pos_2, sizes=None)

        force = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        actions = jnp.array([[0.5, 0.0], [0.0, 0.5]])

        jitted_integrate = jax.jit(dynamic.integrate)
        new_state = jitted_integrate(key, force, state, actions)

        assert new_state.agent_pos.shape == (2, 2)
        assert new_state.agent_vel.shape == (2, 2)
        assert new_state.agent_angle.shape == (2, 1)

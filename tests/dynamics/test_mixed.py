import pytest
import jax
import jax.numpy as jnp

from camar.dynamics import MixedDynamic, HolonomicDynamic, DiffDriveDynamic, HolonomicState, DiffDriveState


class TestMixedDynamic:
    def test_mixed_dynamic_creation(self):
        holonomic = HolonomicDynamic()
        diffdrive = DiffDriveDynamic()

        mixed = MixedDynamic(dynamics_batch=[holonomic, diffdrive], num_agents_batch=[2, 3])

        assert len(mixed.dynamics_batch) == 2
        assert mixed.num_agents_batch == [2, 3]
        assert mixed.num_agents == 5

    def test_mixed_dynamic_creation_single_dynamic(self):
        holonomic = HolonomicDynamic()

        mixed = MixedDynamic(dynamics_batch=[holonomic], num_agents_batch=[4])

        assert len(mixed.dynamics_batch) == 1
        assert mixed.num_agents_batch == [4]
        assert mixed.num_agents == 4

    def test_mixed_dynamic_properties(self):
        holonomic = HolonomicDynamic()
        diffdrive = DiffDriveDynamic()

        mixed = MixedDynamic(dynamics_batch=[holonomic, diffdrive], num_agents_batch=[2, 3])

        assert mixed.action_size == 2  # Both dynamics have action_size 2
        assert mixed.dt == 0.01  # Both dynamics have dt 0.01
        assert mixed.state_class is not None

    def test_mixed_dynamic_invalid_lengths(self):
        holonomic = HolonomicDynamic()

        with pytest.raises(AssertionError, match="dynamics and num_agents must have the same length"):
            MixedDynamic(
                dynamics_batch=[holonomic],
                num_agents_batch=[2, 3],  # Mismatched lengths
            )

    def test_mixed_dynamic_different_dt(self):
        holonomic = HolonomicDynamic(dt=0.01)
        diffdrive = DiffDriveDynamic(dt=0.02)  # Different dt

        with pytest.raises(AssertionError, match="all dynamics must have the same dt"):
            MixedDynamic(dynamics_batch=[holonomic, diffdrive], num_agents_batch=[2, 3])

    def test_mixed_dynamic_state_creation(self):
        holonomic = HolonomicDynamic()
        diffdrive = DiffDriveDynamic()

        mixed = MixedDynamic(dynamics_batch=[holonomic, diffdrive], num_agents_batch=[2, 3])

        key = jax.random.key(0)
        agent_pos = jnp.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],  # First 2 agents (holonomic)
                [2.0, 2.0],
                [3.0, 3.0],
                [4.0, 4.0],  # Next 3 agents (diffdrive)
            ]
        )

        landmark_pos = jnp.zeros((1, 2))
        goal_pos = agent_pos + 1.0
        sizes = {}
        state = mixed.state_class.create(key, landmark_pos, agent_pos, goal_pos, sizes)

        assert state.agent_pos.shape == (5, 2)

        assert hasattr(state, "state_0")  # Holonomic state
        assert state.state_0.agent_pos.shape == (2, 2)
        assert state.state_0.agent_vel.shape == (2, 2)

        assert hasattr(state, "state_1")  # DiffDrive state
        assert state.state_1.agent_pos.shape == (3, 2)
        assert state.state_1.agent_vel.shape == (3, 2)
        assert state.state_1.agent_angle.shape == (3, 1)

    def test_mixed_dynamic_integration_basic(self):
        holonomic = HolonomicDynamic()
        diffdrive = DiffDriveDynamic()

        mixed = MixedDynamic(dynamics_batch=[holonomic, diffdrive], num_agents_batch=[2, 3])

        key = jax.random.key(0)
        agent_pos = jnp.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],  # Holonomic agents
                [2.0, 2.0],
                [3.0, 3.0],
                [4.0, 4.0],  # DiffDrive agents
            ]
        )

        landmark_pos = jnp.zeros((1, 2))
        goal_pos = agent_pos + 1.0
        sizes = {}
        state = mixed.state_class.create(key, landmark_pos, agent_pos, goal_pos, sizes)

        # No forces, no actions
        force = jnp.zeros((5, 2))
        actions = jnp.zeros((5, 2))

        new_state = mixed.integrate(key, force, state, actions)

        assert new_state.agent_pos.shape == (5, 2)

        assert hasattr(new_state, "state_0")
        assert new_state.state_0.agent_pos.shape == (2, 2)
        assert new_state.state_0.agent_vel.shape == (2, 2)

        assert hasattr(new_state, "state_1")
        assert new_state.state_1.agent_pos.shape == (3, 2)
        assert new_state.state_1.agent_vel.shape == (3, 2)
        assert new_state.state_1.agent_angle.shape == (3, 1)

    def test_mixed_dynamic_integration_with_actions(self):
        holonomic = HolonomicDynamic()
        diffdrive = DiffDriveDynamic()

        mixed = MixedDynamic(dynamics_batch=[holonomic, diffdrive], num_agents_batch=[2, 3])

        key = jax.random.key(0)
        agent_pos = jnp.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],  # Holonomic agents
                [2.0, 2.0],
                [3.0, 3.0],
                [4.0, 4.0],  # DiffDrive agents
            ]
        )

        landmark_pos = jnp.zeros((1, 2))
        goal_pos = agent_pos + 1.0
        sizes = {}
        state = mixed.state_class.create(key, landmark_pos, agent_pos, goal_pos, sizes)

        # Different actions for different agent types
        force = jnp.zeros((5, 2))
        actions = jnp.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],  # Holonomic actions
                [0.5, 0.5],
                [0.0, 0.0],
                [1.0, -1.0],  # DiffDrive actions
            ]
        )

        new_state = mixed.integrate(key, force, state, actions)

        assert new_state.agent_pos.shape == (5, 2)
        # Check that positions changed
        assert not jnp.allclose(new_state.agent_pos, state.agent_pos)
        assert not jnp.allclose(new_state.state_0.agent_pos, state.state_0.agent_pos)
        assert not jnp.allclose(new_state.state_1.agent_pos, state.state_1.agent_pos)

    def test_mixed_dynamic_multiple_dynamics(self):
        """Test integration with multiple different dynamics"""
        holonomic1 = HolonomicDynamic(accel=5.0, mass=1.0)
        holonomic2 = HolonomicDynamic(accel=10.0, mass=10.0)
        diffdrive = DiffDriveDynamic(mass=7.0)

        mixed = MixedDynamic(dynamics_batch=[holonomic1, holonomic2, diffdrive], num_agents_batch=[2, 1, 3])

        key = jax.random.key(0)
        agent_pos = jnp.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],  # Holonomic1 agents
                [2.0, 2.0],  # Holonomic2 agent
                [3.0, 3.0],
                [4.0, 4.0],
                [5.0, 5.0],  # DiffDrive agents
            ]
        )

        landmark_pos = jnp.zeros((1, 2))
        goal_pos = agent_pos + 1.0
        sizes = {}
        state = mixed.state_class.create(key, landmark_pos, agent_pos, goal_pos, sizes)

        force = jnp.zeros((6, 2))
        actions = jnp.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],  # Holonomic1 actions
                [0.5, 0.5],  # Holonomic2 action
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],  # DiffDrive actions
            ]
        )

        new_state = mixed.integrate(key, force, state, actions)

        assert new_state.agent_pos.shape == (6, 2)

        assert hasattr(new_state, "state_0")  # Holonomic1
        assert new_state.state_0.agent_pos.shape == (2, 2)
        assert new_state.state_0.agent_vel.shape == (2, 2)

        assert hasattr(new_state, "state_1")  # Holonomic2
        assert new_state.state_1.agent_pos.shape == (1, 2)
        assert new_state.state_1.agent_vel.shape == (1, 2)

        assert hasattr(new_state, "state_2")  # DiffDrive
        assert new_state.state_2.agent_pos.shape == (3, 2)
        assert new_state.state_2.agent_vel.shape == (3, 2)
        assert new_state.state_2.agent_angle.shape == (3, 1)

    def test_mixed_dynamic_jit_compatibility(self):
        # check with different masses
        holonomic = HolonomicDynamic(mass=1.0)
        diffdrive = DiffDriveDynamic(mass=7.0)

        mixed = MixedDynamic(dynamics_batch=[holonomic, diffdrive], num_agents_batch=[2, 3])

        key = jax.random.key(0)
        agent_pos = jnp.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],  # Holonomic agents
                [2.0, 2.0],
                [3.0, 3.0],
                [4.0, 4.0],  # DiffDrive agents
            ]
        )

        landmark_pos = jnp.zeros((1, 2))
        goal_pos = agent_pos + 1.0
        sizes = {}
        state = mixed.state_class.create(key, landmark_pos, agent_pos, goal_pos, sizes)

        force = jnp.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],  # Forces for holonomic agents
                [0.5, 0.5],
                [1.0, 0.0],
                [0.0, 1.0],  # Forces for diffdrive agents
            ]
        )
        actions = jnp.array(
            [
                [0.5, 0.0],
                [0.0, 0.5],  # Actions for holonomic agents
                [0.5, 0.5],
                [1.0, 0.0],
                [0.0, 1.0],  # Actions for diffdrive agents
            ]
        )

        # JIT the integrate function
        jitted_integrate = jax.jit(mixed.integrate)
        new_state = jitted_integrate(key, force, state, actions)

        assert new_state.agent_pos.shape == (5, 2)

        assert hasattr(new_state, "state_0")
        assert new_state.state_0.agent_pos.shape == (2, 2)
        assert new_state.state_0.agent_vel.shape == (2, 2)

        assert hasattr(new_state, "state_1")
        assert new_state.state_1.agent_pos.shape == (3, 2)
        assert new_state.state_1.agent_vel.shape == (3, 2)
        assert new_state.state_1.agent_angle.shape == (3, 1)

    def test_mixed_dynamic_state_access(self):
        holonomic = HolonomicDynamic()
        diffdrive = DiffDriveDynamic()

        mixed = MixedDynamic(dynamics_batch=[holonomic, diffdrive], num_agents_batch=[2, 3])

        key = jax.random.key(0)
        agent_pos = jnp.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],  # Holonomic agents
                [2.0, 2.0],
                [3.0, 3.0],
                [4.0, 4.0],  # DiffDrive agents
            ]
        )

        landmark_pos = jnp.zeros((1, 2))
        goal_pos = agent_pos + 1.0
        sizes = {}
        state = mixed.state_class.create(key, landmark_pos, agent_pos, goal_pos, sizes)

        # Access individual states
        holonomic_state = state.state_0
        diffdrive_state = state.state_1

        assert isinstance(holonomic_state, HolonomicState)
        assert isinstance(diffdrive_state, DiffDriveState)

        assert holonomic_state.agent_pos.shape == (2, 2)
        assert diffdrive_state.agent_pos.shape == (3, 2)
        assert diffdrive_state.agent_angle.shape == (3, 1)

    def test_mixed_dynamic_zero_agents(self):
        holonomic = HolonomicDynamic()

        mixed = MixedDynamic(dynamics_batch=[holonomic], num_agents_batch=[0])

        assert mixed.num_agents == 0

        key = jax.random.key(0)
        agent_pos = jnp.array([]).reshape(0, 2)

        landmark_pos = jnp.zeros((1, 2))
        goal_pos = agent_pos + 1.0
        sizes = {}
        state = mixed.state_class.create(key, landmark_pos, agent_pos, goal_pos, sizes)
        assert state.agent_pos.shape == (0, 2)

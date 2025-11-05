import jax
import jax.numpy as jnp
import pytest

from camar.maps import random_grid


class TestRandomGrid:
    def test_map_creation(self):
        map_gen = random_grid()
        assert map_gen is not None
        assert map_gen.num_agents > 0
        assert map_gen.num_landmarks > 0

    def test_map_reset(self):
        map_gen = random_grid(num_agents=4, num_rows=10, num_cols=10)
        key = jax.random.key(0)
        key_g, landmark_pos, agent_pos, goal_pos, sizes = map_gen.reset(key)

        assert key_g is not None
        assert landmark_pos is not None
        assert agent_pos is not None
        assert goal_pos is not None
        assert sizes is not None

        assert landmark_pos.shape == (map_gen.num_landmarks, 2)
        assert agent_pos.shape == (map_gen.num_agents, 2)
        assert goal_pos.shape == (map_gen.num_agents, 2)

    def test_map_properties(self):
        map_gen = random_grid(obstacle_size=0.5, grain_factor=3)
        assert map_gen.landmark_rad == pytest.approx(0.5 / (2 * (3 - 1)))

    def test_heterogeneous_agents(self):
        """Test heterogeneous agent sizes in random_grid"""
        map_gen = random_grid(num_agents=3, agent_rad_range=(0.02, 0.08))

        assert not map_gen.homogeneous_agents
        assert map_gen.homogeneous_landmarks
        assert not map_gen.homogeneous_goals

        key = jax.random.key(0)
        key_g, landmark_pos, agent_pos, goal_pos, sizes = map_gen.reset(key)

        # Check that agent radii are generated
        assert hasattr(sizes, "agent_rad")
        assert sizes.agent_rad.shape == (3,)
        assert jnp.all(sizes.agent_rad >= 0.02)
        assert jnp.all(sizes.agent_rad <= 0.08)

        # Check that goal radii are proportional to agent radii
        assert hasattr(sizes, "goal_rad")
        assert sizes.goal_rad.shape == (3,)
        assert jnp.allclose(sizes.goal_rad, sizes.agent_rad / 2.5, atol=1e-6)

    def test_heterogeneous_goals(self):
        """Test heterogeneous goal sizes in random_grid"""
        map_gen = random_grid(
            num_agents=2,
            agent_rad_range=(0.03, 0.03),  # Homogeneous agents
            goal_rad_range=(0.01, 0.05),  # Heterogeneous goals
        )

        assert map_gen.homogeneous_agents
        assert map_gen.homogeneous_landmarks
        assert not map_gen.homogeneous_goals

        key = jax.random.key(0)
        key_g, landmark_pos, agent_pos, goal_pos, sizes = map_gen.reset(key)

        # Check that goal radii are generated independently
        assert hasattr(sizes, "goal_rad")
        assert sizes.goal_rad.shape == (2,)
        assert jnp.all(sizes.goal_rad >= 0.01)
        assert jnp.all(sizes.goal_rad <= 0.05)

        # Agent radii should be homogeneous
        assert not hasattr(sizes, "agent_rad")

    def test_heterogeneous_agents_and_goals(self):
        """Test both heterogeneous agents and goals in random_grid"""
        map_gen = random_grid(num_agents=4, agent_rad_range=(0.02, 0.06), goal_rad_range=(0.01, 0.03))

        assert not map_gen.homogeneous_agents
        assert map_gen.homogeneous_landmarks
        assert not map_gen.homogeneous_goals

        key = jax.random.key(0)
        key_g, landmark_pos, agent_pos, goal_pos, sizes = map_gen.reset(key)

        # Check both agent and goal radii
        assert hasattr(sizes, "agent_rad")
        assert hasattr(sizes, "goal_rad")
        assert sizes.agent_rad.shape == (4,)
        assert sizes.goal_rad.shape == (4,)

        # Verify ranges
        assert jnp.all(sizes.agent_rad >= 0.02)
        assert jnp.all(sizes.agent_rad <= 0.06)
        assert jnp.all(sizes.goal_rad >= 0.01)
        assert jnp.all(sizes.goal_rad <= 0.03)

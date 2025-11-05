import jax
import jax.numpy as jnp

from camar.maps import string_grid

MAP_STR = """
.....#.....
.....#.....
...........
.....#.....
.....#.....
"""


class TestStringGrid:
    def test_map_creation(self):
        map_gen = string_grid(map_str=MAP_STR, num_agents=2)
        assert map_gen is not None
        assert map_gen.num_agents == 2
        assert map_gen.num_landmarks == 4 + 11 * 2 + 5 * 2 + 4

    def test_map_reset(self):
        map_gen = string_grid(map_str=MAP_STR, num_agents=2)
        key = jax.random.key(0)
        key_g, landmark_pos, agent_pos, goal_pos, sizes = map_gen.reset(key)

        assert key_g is not None
        assert landmark_pos is not None
        assert agent_pos is not None
        assert goal_pos is not None
        assert sizes is not None

        assert landmark_pos.shape[0] == map_gen.num_landmarks
        assert landmark_pos.shape[1] == 2
        assert agent_pos.shape == (map_gen.num_agents, 2)
        assert goal_pos.shape == (map_gen.num_agents, 2)

    def test_map_creation_with_specific_agent_goal_pos(self):
        map_gen_no_border = string_grid(
            map_str=MAP_STR,
            num_agents=1,
            agent_idx=jnp.array([[0, 0]]),  # Top-left corner
            goal_idx=jnp.array([[2, 5]]),  # A middle free spot
            add_border=False,
            remove_border=False,
        )
        key = jax.random.key(1)
        _, _, agent_pos, goal_pos, sizes = map_gen_no_border.reset(key)

        assert agent_pos is not None
        assert goal_pos is not None
        assert sizes is not None
        assert agent_pos.shape == (1, 2)
        assert goal_pos.shape == (1, 2)

    def test_heterogeneous_agents(self):
        """Test heterogeneous agent sizes"""
        map_gen = string_grid(
            map_str=MAP_STR,
            num_agents=3,
            agent_rad_range=(0.02, 0.08),  # Different agent sizes
        )

        assert not map_gen.homogeneous_agents
        assert map_gen.homogeneous_landmarks
        assert not map_gen.homogeneous_goals  # Goals become heterogeneous when agents are

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
        """Test heterogeneous goal sizes independent of agents"""
        map_gen = string_grid(
            map_str=MAP_STR,
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
        """Test both heterogeneous agents and goals"""
        map_gen = string_grid(
            map_str=MAP_STR, num_agents=4, agent_rad_range=(0.02, 0.06), goal_rad_range=(0.01, 0.03)
        )

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

    def test_homogeneous_configuration(self):
        """Test that homogeneous configuration works correctly"""
        map_gen = string_grid(
            map_str=MAP_STR,
            num_agents=2,
            agent_rad_range=(0.03, 0.03),  # Same min/max = homogeneous
            goal_rad_range=(0.01, 0.01),  # Same min/max = homogeneous
        )

        assert map_gen.homogeneous_agents
        assert map_gen.homogeneous_landmarks
        assert map_gen.homogeneous_goals

        key = jax.random.key(0)
        key_g, landmark_pos, agent_pos, goal_pos, sizes = map_gen.reset(key)

        # Should not have agent_rad or goal_rad in sizes
        assert not hasattr(sizes, "agent_rad")
        assert not hasattr(sizes, "goal_rad")

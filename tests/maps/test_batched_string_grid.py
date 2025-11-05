import jax
import jax.numpy as jnp

from camar.maps import batched_string_grid

MAP_STR_1 = """
.....#.....
.....#.....
...........
...........
.....#.....
.....#.....
"""

MAP_STR_2 = """
...........
..#######..
..#.....#..
..#.....#..
..#######..
...........
"""

MAP_STR_BATCH = [MAP_STR_1, MAP_STR_2]

MAP_STR_SQUARE_1 = """
.....
.###.
.#.#.
.###.
.....
"""  # 5x5

MAP_STR_SQUARE_2 = """
.#.#.
#.#.#
.#.#.
#.#.#
.#.#.
"""  # 5x5

MAP_STR_SQUARE_BATCH = [MAP_STR_SQUARE_1, MAP_STR_SQUARE_2]


class TestBatchedStringGrid:
    def test_map_creation(self):
        map_gen = batched_string_grid(map_str_batch=MAP_STR_BATCH, num_agents=2)
        assert map_gen is not None
        assert map_gen.num_agents == 2
        assert map_gen.batch_size == len(MAP_STR_SQUARE_BATCH)
        assert map_gen.num_landmarks > 0

    def test_map_reset(self):
        num_agents = 2
        map_gen = batched_string_grid(map_str_batch=MAP_STR_BATCH, num_agents=num_agents)
        key = jax.random.key(0)

        keys_g, landmark_pos, agent_pos, goal_pos, sizes = map_gen.reset(key)

        assert landmark_pos is not None
        assert agent_pos is not None
        assert goal_pos is not None
        assert sizes is not None

        assert landmark_pos.shape == (map_gen.num_landmarks, 2)
        assert agent_pos.shape == (num_agents, 2)
        assert goal_pos.shape == (num_agents, 2)

    def test_map_creation_with_specific_agent_goal_pos(self):
        agent_idx_batch = [
            jnp.array([[2, 2]]),  # For MAP_STR_SQUARE_1
            jnp.array([[0, 0]]),  # For MAP_STR_SQUARE_2
        ]
        goal_idx_batch = [
            jnp.array([[0, 0]]),
            jnp.array([[0, 2]]),
        ]

        map_gen = batched_string_grid(
            map_str_batch=MAP_STR_SQUARE_BATCH,
            num_agents=1,
            agent_idx_batch=agent_idx_batch,
            goal_idx_batch=goal_idx_batch,
            add_border=False,
            remove_border=False,
        )
        key = jax.random.key(1)
        _, _, agent_pos, goal_pos, sizes = map_gen.reset(key)

        assert agent_pos is not None
        assert goal_pos is not None
        assert sizes is not None
        assert agent_pos.shape == (1, 2)
        assert goal_pos.shape == (1, 2)

    def test_heterogeneous_agents(self):
        """Test heterogeneous agent sizes in batched_string_grid"""
        map_gen = batched_string_grid(map_str_batch=MAP_STR_BATCH, num_agents=3, agent_rad_range=(0.02, 0.08))

        assert not map_gen.homogeneous_agents
        assert map_gen.homogeneous_landmarks
        assert not map_gen.homogeneous_goals

        key = jax.random.key(0)
        keys_g, landmark_pos, agent_pos, goal_pos, sizes = map_gen.reset(key)

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
        """Test heterogeneous goal sizes in batched_string_grid"""
        map_gen = batched_string_grid(
            map_str_batch=MAP_STR_BATCH,
            num_agents=2,
            agent_rad_range=(0.03, 0.03),  # Homogeneous agents
            goal_rad_range=(0.01, 0.05),  # Heterogeneous goals
        )

        assert map_gen.homogeneous_agents
        assert map_gen.homogeneous_landmarks
        assert not map_gen.homogeneous_goals

        key = jax.random.key(0)
        keys_g, landmark_pos, agent_pos, goal_pos, sizes = map_gen.reset(key)

        # Check that goal radii are generated independently
        assert hasattr(sizes, "goal_rad")
        assert sizes.goal_rad.shape == (2,)
        assert jnp.all(sizes.goal_rad >= 0.01)
        assert jnp.all(sizes.goal_rad <= 0.05)

        # Agent radii should be homogeneous
        assert not hasattr(sizes, "agent_rad")

    def test_heterogeneous_agents_and_goals(self):
        """Test both heterogeneous agents and goals in batched_string_grid"""
        map_gen = batched_string_grid(
            map_str_batch=MAP_STR_BATCH,
            num_agents=4,
            agent_rad_range=(0.02, 0.06),
            goal_rad_range=(0.01, 0.03),
        )

        assert not map_gen.homogeneous_agents
        assert map_gen.homogeneous_landmarks
        assert not map_gen.homogeneous_goals

        key = jax.random.key(0)
        keys_g, landmark_pos, agent_pos, goal_pos, sizes = map_gen.reset(key)

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

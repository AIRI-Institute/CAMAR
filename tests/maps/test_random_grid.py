import jax
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
        key_g, landmark_pos, agent_pos, goal_pos = map_gen.reset(key)

        assert key_g is not None
        assert landmark_pos is not None
        assert agent_pos is not None
        assert goal_pos is not None

        assert landmark_pos.shape == (map_gen.num_landmarks, 2)
        assert agent_pos.shape == (map_gen.num_agents, 2)
        assert goal_pos.shape == (map_gen.num_agents, 2)

    def test_map_properties(self):
        map_gen = random_grid(obstacle_size=0.5, grain_factor=3)
        assert map_gen.landmark_rad == pytest.approx(0.5 / (2 * (3 - 1)))

import jax
import pytest

from camar.maps import caves_cont


class TestCavesCont:
    def test_map_creation(self):
        map_gen = caves_cont()
        assert map_gen is not None
        assert map_gen.num_agents > 0
        assert map_gen.num_landmarks > 0

        map_gen_no_border = caves_cont(add_borders=False)
        assert map_gen_no_border is not None
        assert map_gen_no_border.border_landmarks.shape[0] == 0

    def test_map_reset(self):
        num_agents = 4
        map_gen = caves_cont(
            num_agents=num_agents, num_rows=32, num_cols=32, scale=7, free_ratio=0.5
        )
        key = jax.random.key(0)
        key_g, landmark_pos, agent_pos, goal_pos = map_gen.reset(key)

        assert key_g is not None
        assert landmark_pos is not None
        assert agent_pos is not None
        assert goal_pos is not None

        assert landmark_pos.shape == (map_gen.num_landmarks, 2)
        assert agent_pos.shape == (num_agents, 2)
        assert goal_pos.shape == (num_agents, 2)

    def test_map_properties(self):
        obstacle_size = 0.2
        agent_size = 0.1
        map_gen = caves_cont(obstacle_size=obstacle_size, agent_size=agent_size)
        assert map_gen.landmark_rad == pytest.approx(obstacle_size / 2)
        assert map_gen.agent_rad == pytest.approx(agent_size / 2)
        assert map_gen.goal_rad == pytest.approx((agent_size / 2) / 2.5)
        assert map_gen.height == map_gen.num_cols * obstacle_size
        assert map_gen.width == map_gen.num_rows * obstacle_size

    def test_invalid_landmark_ratios(self):
        with pytest.raises(ValueError):
            caves_cont(landmark_low_ratio=0.8, landmark_high_ratio=0.7)

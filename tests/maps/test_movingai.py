import jax
import pytest

from camar.maps import movingai

TEST_MAP_NAMES = ["street/Denver_0_1024"]


class TestMovingAI:
    def test_map_creation(self):
        # try:
        map_gen = movingai(map_names=TEST_MAP_NAMES, num_agents=2)
        assert map_gen is not None
        assert map_gen.num_agents == 2
        assert map_gen.batch_size == len(TEST_MAP_NAMES)
        assert map_gen.num_landmarks > 0

    # except Exception as e:
    #     pytest.skip(
    #         f"Skipping MovingAI test due to potential network/data issue: {e}"
    #     )

    def test_map_reset(self):
        # try:
        num_agents = 2
        map_gen = movingai(map_names=TEST_MAP_NAMES, num_agents=num_agents)
        key = jax.random.key(0)

        keys_g, landmark_pos, agent_pos, goal_pos = map_gen.reset(key)

        assert landmark_pos is not None
        assert agent_pos is not None
        assert goal_pos is not None

        assert landmark_pos.shape == (map_gen.num_landmarks, 2)
        assert agent_pos.shape == (num_agents, 2)
        assert goal_pos.shape == (num_agents, 2)

    # except Exception as e:
    #     pytest.skip(
    #         f"Skipping MovingAI test due to potential network/data issue: {e}"
    #     )

    def test_map_properties(self):
        # try:
        obstacle_size = 0.15
        agent_size = 0.07
        map_gen = movingai(
            map_names=TEST_MAP_NAMES,
            obstacle_size=obstacle_size,
            agent_size=agent_size,
        )

        assert map_gen.landmark_rad == pytest.approx(obstacle_size / 2)
        assert map_gen.agent_rad == pytest.approx(agent_size / 2)
        assert map_gen.goal_rad == pytest.approx((agent_size / 2) / 2.5)

        resized_h, resized_w = 128, 128

        assert map_gen.height == pytest.approx((resized_h - 2) * obstacle_size)
        assert map_gen.width == pytest.approx((resized_w - 2) * obstacle_size)

    # except Exception as e:
    #     pytest.skip(
    #         f"Skipping MovingAI test due to potential network/data issue: {e}"
    #     )

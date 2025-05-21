import jax
import pytest

from camar.maps import labmaze_grid


class TestLabMazeGrid:
    def test_map_creation(self):
        num_maps = 2
        map_gen = labmaze_grid(num_maps=num_maps, num_agents=4, height=7, width=7)
        assert map_gen is not None
        assert map_gen.num_agents == 4
        assert map_gen.batch_size == num_maps
        assert map_gen.num_landmarks > 0

    def test_map_reset(self):
        num_maps = 2
        num_agents = 3
        map_gen = labmaze_grid(
            num_maps=num_maps,
            num_agents=num_agents,
            height=7,
            width=7,
            room_min_size=2,
            room_max_size=3,
        )
        key = jax.random.key(0)

        keys_g, landmark_pos, agent_pos, goal_pos = map_gen.reset(key)

        assert keys_g is not None
        assert landmark_pos is not None
        assert agent_pos is not None
        assert goal_pos is not None

        assert landmark_pos.shape == (map_gen.num_landmarks, 2)
        assert agent_pos.shape == (num_agents, 2)
        assert goal_pos.shape == (num_agents, 2)

    def test_map_properties(self):
        obstacle_size = 0.25
        agent_size = 0.12
        map_gen = labmaze_grid(
            num_maps=1,
            obstacle_size=obstacle_size,
            agent_size=agent_size,
            height=7,
            width=7,
            room_min_size=2,
            room_max_size=3,
        )

        assert map_gen.landmark_rad == pytest.approx(obstacle_size / 2)
        assert map_gen.agent_rad == pytest.approx(agent_size / 2)
        assert map_gen.goal_rad == pytest.approx((agent_size / 2) / 2.5)

        init_height, init_width = 7, 7
        map_gen_dims = labmaze_grid(
            num_maps=1,
            height=init_height,
            width=init_width,
            obstacle_size=obstacle_size,
        )
        assert map_gen_dims.height == pytest.approx(init_height * obstacle_size)
        assert map_gen_dims.width == pytest.approx(init_width * obstacle_size)

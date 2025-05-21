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
        key_g, landmark_pos, agent_pos, goal_pos = map_gen.reset(key)

        assert key_g is not None
        assert landmark_pos is not None
        assert agent_pos is not None
        assert goal_pos is not None

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
        _, _, agent_pos, goal_pos = map_gen_no_border.reset(key)

        assert agent_pos is not None
        assert goal_pos is not None
        assert agent_pos.shape == (1, 2)
        assert goal_pos.shape == (1, 2)

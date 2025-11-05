from typing import Optional, Tuple

from .base import base_map
from .batched_string_grid import batched_string_grid
from camar.registry import register_map


def generate_labmaze_maps(num_maps, height, width, max_rooms, seed, **labmaze_kwargs):
    from labmaze import RandomMaze

    maps = []
    free_pos = []
    for i in range(num_maps):
        random_maze = RandomMaze(
            height=height,
            width=width,
            max_rooms=max_rooms,
            random_seed=seed + i,
            **labmaze_kwargs,
        )
        maps.append(str(random_maze.entity_layer))
        free_pos.append(str(random_maze.variations_layer).replace(".", "*"))
    return maps, free_pos


@register_map()
class labmaze_grid(batched_string_grid):
    def __init__(
        self,
        num_maps: int,
        height: int = 11,
        width: int = 11,
        max_rooms: int = -1,
        seed: int = 0,
        num_agents: int = 10,
        landmark_rad: float = 0.1,
        agent_rad_range: Optional[Tuple[float, float]] = (0.05, 0.05),
        goal_rad_range: Optional[Tuple[float, float]] = None,
        max_free_pos: Optional[int] = None,
        **labmaze_kwargs,
    ) -> base_map:
        map_str_batch, free_pos_str_batch = generate_labmaze_maps(
            num_maps=num_maps,
            height=height,
            width=width,
            max_rooms=max_rooms,
            seed=seed,
            **labmaze_kwargs,
        )

        super().__init__(
            map_str_batch=map_str_batch,
            free_pos_str_batch=free_pos_str_batch,
            agent_idx_batch=None,
            goal_idx_batch=None,
            num_agents=num_agents,
            random_agents=True,
            random_goals=True,
            remove_border=False,
            add_border=False,
            landmark_rad=landmark_rad,
            agent_rad_range=agent_rad_range,
            goal_rad_range=goal_rad_range,
            max_free_pos=max_free_pos,
        )

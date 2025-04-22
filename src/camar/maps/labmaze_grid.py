import labmaze

from .base_map import base_map
from .batched_string_grid import batched_string_grid


def generate_labmaze_maps(num_maps, height, width, max_rooms, seed, **labmaze_kwargs):
    maps = []
    for i in range(num_maps):
        random_maze = labmaze.RandomMaze(
            height=height,
            width=width,
            max_rooms=max_rooms,
            random_seed=seed + i,
            **labmaze_kwargs,
        )
        maps.append(str(random_maze.entity_layer))
    return maps


class labmaze_grid(batched_string_grid):
    def __init__(
        self,
        num_maps: int,
        height: int = 11,
        width: int = 11,
        max_rooms: int = -1,
        seed: int = 0,
        num_agents: int = 10,
        obstacle_size: float = 0.1,
        agent_size: float = 0.06,
        **labmaze_kwargs,
    ) -> base_map:
        map_str_batch = generate_labmaze_maps(
            num_maps=num_maps,
            height=height,
            width=width,
            max_rooms=max_rooms,
            seed=seed,
            **labmaze_kwargs,
        )

        super().__init__(
            map_str_batch=map_str_batch,
            agent_idx_batch=None,
            goal_idx_batch=None,
            num_agents=num_agents,
            random_agents=True,
            random_goals=True,
            remove_border=False,
            add_border=False,
            obstacle_size=obstacle_size,
            agent_size=agent_size,
        )

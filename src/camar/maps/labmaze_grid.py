import labmaze

from camar.maps.string_grid import string_grid


class labmaze_grid(string_grid):
    def __init__(
        self,
        num_agents: int = 10,
        obstacle_size: float = 0.1,
        agent_size: float = 0.09,
        **kwargs,
    ):
        maze = labmaze.RandomMaze(**kwargs)
        map_str = str(maze.entity_layer)

        super().__init__(
            map_str,
            num_agents,
            obstacle_size,
            agent_size,
            remove_border=False,
            add_border=False,
        )

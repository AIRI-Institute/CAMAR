import labmaze

from camar.core.maps.grid import Grid


class LabMaze(Grid):
    def __init__(
        self,
        num_agents: int = 10,
        obstacle_size: float = 0.1,
        agent_size: float = 0.09,
        **kwargs,
    ):
        maze = labmaze.RandomMaze(**kwargs)
        map_str = str(maze.entity_layer).replace(" ", ".").replace("*", "#")

        super().__init__(map_str, num_agents, obstacle_size, agent_size, remove_border = False, add_border = False)

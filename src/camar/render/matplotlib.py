from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, ConnectionPatch

from .const import COLORS, LANDMARK_COLOR


@dataclass
class MPLRenderConfig:
    name: str
    color: str
    alpha: float = 1.0
    use_index_colors: bool = False
    show_connections: bool = False


class MPLVisualizer:
    def __init__(
        self,
        env,
        state_seq: list,
        reward_seq=None,
        path=None,
        fps: Optional[int] = None,
        use_all_colors: bool = False,
        show_connections: bool = True,
        update_radii: bool = False,
    ):
        self.env = env
        self.state_seq = state_seq
        self.reward_seq = reward_seq
        self.path = path
        self.fps = fps or (8 / env.step_dt)
        self.use_all_colors = use_all_colors
        self.show_connections = show_connections
        self.update_radii = update_radii

        # Setup render configurations
        self._setup_render_configs()

        # Initialize the plot
        self._init_render()

    def _setup_render_configs(self):
        """Setup render configurations for different object types."""
        self.render_configs = {
            "landmarks": MPLRenderConfig(
                name="landmarks", color=LANDMARK_COLOR, alpha=1.0, use_index_colors=False
            ),
            "goals": MPLRenderConfig(
                name="goals",
                color="blue",
                alpha=1.0,
                use_index_colors=True,
                show_connections=False,
            ),
            "agents": MPLRenderConfig(
                name="agents",
                color="dodgerblue",
                alpha=0.8,
                use_index_colors=True,
                show_connections=self.show_connections,
            ),
        }

    def _get_color(self, config: MPLRenderConfig, index: int) -> str:
        """Generate color for an object based on configuration."""
        if config.use_index_colors and self.use_all_colors:
            colors = plt.cm.Set3(np.linspace(0, 1, 12))
            return colors[index % len(colors)]
        elif config.use_index_colors:
            hex_color = COLORS[index % len(COLORS)]
            hex_color = hex_color.lstrip("#")
            return tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
        else:
            return config.color

    def _get_radius(self, state, data_type: str, index: int) -> float:
        """Get radius for an object, handling both homogeneous and heterogeneous cases."""
        if data_type == "landmarks":
            if self.env.homogeneous_landmarks:
                return self.env.map_generator.landmark_rad
            else:
                return state.sizes.landmark_rad[index]
        elif data_type == "goals":
            if self.env.homogeneous_goals:
                return self.env.map_generator.goal_rad
            else:
                return state.sizes.goal_rad[index]
        elif data_type == "agents":
            if self.env.homogeneous_agents:
                return self.env.map_generator.agent_rad
            else:
                return state.sizes.agent_rad[index]
        else:
            raise ValueError(f"Unknown data type: {data_type}")

    def _get_positions(self, state, data_type: str) -> List[Tuple[float, float]]:
        """Get positions for objects of a specific type."""
        if data_type == "landmarks":
            return state.landmark_pos.tolist()
        elif data_type == "goals":
            return state.goal_pos.tolist()
        elif data_type == "agents":
            return state.physical_state.agent_pos.tolist()
        else:
            raise ValueError(f"Unknown data type: {data_type}")

    def _init_render(self):
        """Initialize the matplotlib figure and artists."""
        state = self.state_seq[0]

        # Create figure based on environment dimensions
        fig_width = max(15.0, min(30.0, self.env.width / 20))  # Comfortable range
        fig_height = max(15.0, min(30.0, self.env.height / 20))
        self.fig, self.ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))

        # Plot path if provided
        if self.path is not None:
            self.ax.plot(self.path[:, 0], -self.path[:, 1], lw=1, alpha=0.7)  # invert Y

        # Set plot limits
        xlim = (self.env.width + 1) / 2
        ylim = (self.env.height + 1) / 2
        self.ax.set_xlim([-xlim, xlim])
        self.ax.set_ylim([-ylim, ylim])

        # Set equal aspect ratio to prevent different scaling
        self.ax.set_aspect("equal")

        # Initialize artist collections
        self.artists = {"landmarks": [], "goals": [], "agents": [], "connections": []}

        # Create initial artists
        self._create_artists(state)

        # Set initial title
        self.ax.set_title(f"Step: {state.step}")

    def _create_artists(self, state):
        """Create all artists for a given state."""
        # Clear existing artists
        for artist_list in self.artists.values():
            for artist in artist_list:
                artist.remove()
            artist_list.clear()

        # Create landmarks
        config = self.render_configs["landmarks"]
        positions = self._get_positions(state, "landmarks")
        for i, (x, y) in enumerate(positions):
            radius = self._get_radius(state, "landmarks", i)
            color = self._get_color(config, i)
            circle = Circle((x, -y), radius, color=color, alpha=config.alpha)  # invert Y
            self.ax.add_patch(circle)
            self.artists["landmarks"].append(circle)

        # Create goals
        config = self.render_configs["goals"]
        positions = self._get_positions(state, "goals")
        for i, (x, y) in enumerate(positions):
            radius = self._get_radius(state, "goals", i)
            color = self._get_color(config, i)
            circle = Circle((x, -y), radius, color=color, alpha=config.alpha)  # invert Y
            self.ax.add_patch(circle)
            self.artists["goals"].append(circle)

        # Create agents
        config = self.render_configs["agents"]
        positions = self._get_positions(state, "agents")
        for i, (x, y) in enumerate(positions):
            radius = self._get_radius(state, "agents", i)
            color = self._get_color(config, i)
            circle = Circle((x, -y), radius, color=color, alpha=config.alpha)  # invert Y
            self.ax.add_patch(circle)
            self.artists["agents"].append(circle)

        # Create connections if enabled
        if config.show_connections:
            agent_positions = self._get_positions(state, "agents")
            goal_positions = self._get_positions(state, "goals")
            for agent_pos, goal_pos in zip(agent_positions, goal_positions):
                # Invert Y coordinates for connections
                agent_pos_inverted = (agent_pos[0], -agent_pos[1])
                goal_pos_inverted = (goal_pos[0], -goal_pos[1])
                connection = ConnectionPatch(
                    agent_pos_inverted,
                    goal_pos_inverted,
                    "data",
                    "data",
                    lw=0.3,
                    ls="--",
                    color="dimgray",
                    alpha=0.8,
                )
                self.ax.add_patch(connection)
                self.artists["connections"].append(connection)

    def _update_artists(self, state):
        """Update artist positions for a new state."""
        # Update agent positions
        agent_positions = self._get_positions(state, "agents")
        for i, circle in enumerate(self.artists["agents"]):
            if i < len(agent_positions):
                # Invert Y coordinate for agent positions
                agent_pos = agent_positions[i]
                circle.center = (agent_pos[0], -agent_pos[1])

        # Update connection positions
        if self.artists["connections"]:
            goal_positions = self._get_positions(state, "goals")
            for i, connection in enumerate(self.artists["connections"]):
                if i < len(agent_positions) and i < len(goal_positions):
                    # Invert Y coordinates for connections
                    agent_pos = agent_positions[i]
                    goal_pos = goal_positions[i]
                    connection.xy1 = (agent_pos[0], -agent_pos[1])
                    connection.xy2 = (goal_pos[0], -goal_pos[1])

        # Update title
        if hasattr(state, "step"):
            self.ax.set_title(f"Step: {state.step}")

    def _update_circle_radii(self, state):
        """Update circle radii if they can change dynamically."""
        try:
            # Update landmark radii
            for i, circle in enumerate(self.artists["landmarks"]):
                if i < len(self._get_positions(state, "landmarks")):
                    radius = self._get_radius(state, "landmarks", i)
                    circle.radius = radius

            # Update goal radii
            for i, circle in enumerate(self.artists["goals"]):
                if i < len(self._get_positions(state, "goals")):
                    radius = self._get_radius(state, "goals", i)
                    circle.radius = radius

            # Update agent radii
            for i, circle in enumerate(self.artists["agents"]):
                if i < len(self._get_positions(state, "agents")):
                    radius = self._get_radius(state, "agents", i)
                    circle.radius = radius

        except Exception as e:
            print(f"Warning: Error updating circle radii: {e}")
            pass

    def animate(self, save_fname: Optional[str] = None, view: bool = True):
        """Create and run the animation."""
        ani = animation.FuncAnimation(
            self.fig,
            self._update_animation,
            frames=len(self.state_seq),
            blit=False,
            interval=1000 / self.fps,
        )

        # Save the animation if filename provided
        if save_fname is not None:
            ani.save(save_fname)

        # Show the animation if requested
        if view:
            plt.show(block=True)

        return ani

    def _update_animation(self, frame):
        """Update function for matplotlib animation."""
        state = self.state_seq[frame]
        self._update_artists(state)

        # Update radii if enabled
        if self.update_radii:
            self._update_circle_radii(state)

        return []

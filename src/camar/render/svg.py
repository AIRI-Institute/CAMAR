import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .const import COLORS, LANDMARK_COLOR
from .utils import hex_to_hsl


@dataclass
class RenderConfig:
    """Configuration for rendering different object types."""

    name: str
    css_class: str
    color: str
    use_index_colors: bool = False
    use_transparency: bool = False
    transparency: float = 1.0
    stroke_config: Optional[str] = None


class CircleRenderer:
    """Handles rendering of animated and static circles."""

    def __init__(
        self,
        scale: float,
        duration: Optional[float] = None,
        keytimes: Optional[str] = None,
    ):
        self.scale = scale
        self.duration = duration
        self.keytimes = keytimes

    def _get_color(self, config: RenderConfig, index: int, use_all_colors: bool, color_step: int) -> str:
        """Generate color for an object based on configuration."""
        if config.use_index_colors and use_all_colors:
            hue = (index * color_step) % 360
            color = f"hsl({hue}, 100%, 50%)"
        elif config.use_index_colors:
            hex_color = COLORS[index % len(COLORS)]
            hue, saturation, lightness = hex_to_hsl(hex_color)
            color = f"hsl({hue}, {saturation}%, {lightness}%)"
        else:
            hex_color = config.color
            hue, saturation, lightness = hex_to_hsl(hex_color)
            color = f"hsl({hue}, {saturation}%, {lightness}%)"

        if config.use_transparency:
            color = color.replace("hsl(", "hsla(").replace("%)", f"%, {config.transparency})")

        return color

    def _create_animated_circle(
        self,
        config: RenderConfig,
        positions: List[Tuple[float, float]],
        radii: Optional[List[float]] = None,
        index: int = 0,
        use_all_colors: bool = False,
        color_step: int = 25,
        homogeneous_radius: Optional[float] = None,
    ) -> str:
        """Create an animated circle with position and radius animations."""
        color = self._get_color(config, index, use_all_colors, color_step)

        # Prepare animation data
        cx_values = [x / self.scale for x, _ in positions]
        cy_values = [y / self.scale for _, y in positions]

        # Build circle element
        circle_attrs = [f'class="{config.css_class}"', f'fill="{color}"']

        if radii is not None:
            r_values = [r / self.scale for r in radii]
            circle_attrs.append(f'r="{r_values[0]:.3f}"')
        elif homogeneous_radius is not None:
            r = homogeneous_radius / self.scale
            circle_attrs.append(f'r="{r:.3f}"')
        else:
            # Fallback - no radius specified
            pass

        circle_start = f"<circle {' '.join(circle_attrs)}>"

        # Add animations
        animations = []
        for attr, values in [("cx", cx_values), ("cy", cy_values)]:
            animations.append(self._create_animation(attr, values))

        if radii is not None:
            animations.append(self._create_animation("r", r_values))

        return circle_start + "".join(animations) + "</circle>"

    def _create_static_circle(
        self,
        config: RenderConfig,
        x: float,
        y: float,
        radius: float,
        index: int = 0,
        use_all_colors: bool = False,
        color_step: int = 25,
    ) -> str:
        """Create a static circle."""
        color = self._get_color(config, index, use_all_colors, color_step)

        cx = x / self.scale
        cy = y / self.scale
        r = radius / self.scale

        return f'<circle class="{config.css_class}" cx="{cx:.3f}" cy="{cy:.3f}" r="{r:.3f}" fill="{color}"></circle>'

    def _create_animation(self, attribute: str, values: List[float]) -> str:
        """Create an SVG animation element."""
        values_str = ";".join(f"{v:.3f}" for v in values)
        return (
            f'<animate attributeName="{attribute}" dur="{self.duration}s" '
            f'keyTimes="{self.keytimes}" repeatCount="indefinite" values="{values_str}"/>'
        )


class SVGVisualizer:
    def __init__(
        self,
        env,
        state_seq,
        animate_agents: bool = True,
        animate_goals: bool = True,
        animate_landmarks: bool = True,
        fps: Optional[float] = None,
        color_step: Optional[int] = None,
        use_all_colors: bool = False,
        agent_transparency: float = 0.8,
    ):
        self.env = env
        self.state_seq = state_seq
        self.use_all_colors = use_all_colors
        self.agent_transparency = agent_transparency

        # Determine if we should animate
        should_animate = isinstance(state_seq, list) and len(state_seq) > 1
        self.animate_agents = should_animate and animate_agents
        self.animate_goals = should_animate and animate_goals
        self.animate_landmarks = should_animate and animate_landmarks

        # Setup timing
        self.fps = fps or (8 / env.step_dt)
        if should_animate:
            self.keytimes = self._create_keytimes()
            self.duration = round(len(state_seq) / self.fps, 3)
        else:
            self.keytimes = None
            self.duration = None

        # Setup rendering
        self.width = env.width
        self.height = env.height
        self.scale = max(self.width, self.height) / 512
        self.color_step = color_step or max(360 // env.num_agents, 25)

        # Initialize renderer
        self.renderer = CircleRenderer(self.scale, self.duration, self.keytimes)

        # Define render configurations
        self._setup_render_configs()

    def _create_keytimes(self) -> str:
        """Create keytimes string for animations."""
        keytimes = [round(i / len(self.state_seq), 8) for i in range(len(self.state_seq) - 1)]
        keytimes.append(1.0)
        return ";".join(map(str, keytimes))

    def _setup_render_configs(self):
        """Setup render configurations for different object types."""
        self.render_configs = {
            "landmarks": RenderConfig(
                name="landmarks",
                css_class="landmark",
                color=LANDMARK_COLOR,
                use_index_colors=False,
                use_transparency=False,
            ),
            "goals": RenderConfig(
                name="goals",
                css_class="goal",
                color=COLORS[0],  # Will be overridden by index
                use_index_colors=True,
                use_transparency=False,
                stroke_config=LANDMARK_COLOR,
            ),
            "agents": RenderConfig(
                name="agents",
                css_class="agent",
                color=COLORS[0],  # Will be overridden by index
                use_index_colors=True,
                use_transparency=True,
                transparency=self.agent_transparency,
            ),
        }

    def _get_state_data(
        self, state, data_type: str
    ) -> Tuple[List[Tuple[float, float]], Optional[List[float]]]:
        """Extract position and radius data from state."""
        if data_type == "landmarks":
            positions = state.landmark_pos.tolist()
            radii = None if self.env.homogeneous_landmarks else state.sizes.landmark_rad.tolist()
        elif data_type == "goals":
            positions = state.goal_pos.tolist()
            radii = None if self.env.homogeneous_goals else state.sizes.goal_rad.tolist()
        elif data_type == "agents":
            positions = state.physical_state.agent_pos.tolist()
            radii = None if self.env.homogeneous_agents else state.sizes.agent_rad.tolist()
        else:
            raise ValueError(f"Unknown data type: {data_type}")

        return positions, radii

    def _get_homogeneous_radius(self, data_type: str) -> float:
        """Get radius for homogeneous objects."""
        if data_type == "landmarks":
            return self.env.map_generator.landmark_rad
        elif data_type == "goals":
            return self.env.map_generator.goal_rad
        elif data_type == "agents":
            return self.env.map_generator.agent_rad
        else:
            raise ValueError(f"Unknown data type: {data_type}")

    def _render_animated_objects(self, data_type: str) -> str:
        """Render animated objects of a specific type."""
        config = self.render_configs[data_type]
        animate_flag = getattr(self, f"animate_{data_type}")

        if not animate_flag:
            return self._render_static_objects(data_type)

        # Collect animation data
        object_dict = {}
        for state in self.state_seq:
            positions, radii = self._get_state_data(state, data_type)

            for obj_i, (x, y) in enumerate(positions):
                if obj_i not in object_dict:
                    object_dict[obj_i] = {"positions": [], "radii": []}

                object_dict[obj_i]["positions"].append((x, y))
                if radii is not None:
                    object_dict[obj_i]["radii"].append(radii[obj_i])

        # Generate SVG for each object
        svg_parts = []
        for obj_i, obj_data in object_dict.items():
            positions = obj_data["positions"]
            radii = obj_data["radii"] if obj_data["radii"] else None

            # Get homogeneous radius if needed
            homogeneous_radius = None
            if radii is None:
                homogeneous_radius = self._get_homogeneous_radius(data_type)

            svg = self.renderer._create_animated_circle(
                config,
                positions,
                radii,
                obj_i,
                self.use_all_colors,
                self.color_step,
                homogeneous_radius,
            )
            svg_parts.append(svg)

        return "\n".join(svg_parts)

    def _render_static_objects(self, data_type: str) -> str:
        """Render static objects of a specific type."""
        config = self.render_configs[data_type]

        # Get the state to render
        if isinstance(self.state_seq, list):
            state = self.state_seq[0]
        else:
            state = self.state_seq

        positions, radii = self._get_state_data(state, data_type)

        svg_parts = []
        for i, (x, y) in enumerate(positions):
            if radii is not None:
                radius = radii[i]
            else:
                radius = self._get_homogeneous_radius(data_type)

            svg = self.renderer._create_static_circle(
                config, x, y, radius, i, self.use_all_colors, self.color_step
            )
            svg_parts.append(svg)

        return "\n".join(svg_parts)

    def _create_svg_header(self) -> str:
        """Create the SVG header with proper dimensions and viewBox."""
        scaled_width = math.ceil(self.width / self.scale)
        scaled_height = math.ceil(self.height / self.scale)

        view_box = (
            -self.width / 2 / self.scale,
            -self.height / 2 / self.scale,
            self.width / self.scale,
            self.height / self.scale,
        )

        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"\n'
            f'\twidth="{scaled_width}" height="{scaled_height}" '
            f'viewBox="{" ".join(map(str, view_box))}">'
        )

    def _create_styles(self) -> str:
        """Create CSS styles for the SVG."""
        goal_stroke = (
            f"stroke: {LANDMARK_COLOR}; stroke-width: {self.env.map_generator.goal_rad / 2 / self.scale};"
            if self.env.homogeneous_goals
            else f"stroke: {LANDMARK_COLOR};"
        )

        return (
            "<defs>\n"
            "<style>\n"
            "\t.landmark { }\n"
            "\t.agent { }\n"
            f"\t.goal {{ {goal_stroke} }}\n"
            "</style>\n"
            "</defs>"
        )

    def render(self) -> str:
        """Render the complete SVG."""
        header = self._create_svg_header()
        styles = self._create_styles()

        # Render all object types
        landmarks_svg = self._render_animated_objects("landmarks")
        goals_svg = self._render_animated_objects("goals")
        agents_svg = self._render_animated_objects("agents")

        # Combine all parts
        parts = [
            header,
            styles,
            "\n",
            landmarks_svg,
            "\n",
            goals_svg,
            "\n",
            agents_svg,
            "</svg>",
        ]

        return "\n".join(parts)

    def save_svg(self, filename: str = "test.svg") -> None:
        """Save the rendered SVG to a file."""
        with open(filename, "w") as svg_file:
            svg_file.write(self.render())

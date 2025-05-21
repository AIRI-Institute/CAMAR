import math


class Visualizer:
    def __init__(
        self,
        env,
        state_seq,
        animate_agents=True,
        animate_goals=True,
        animate_landmarks=True,
        fps=None,
        color_step=None,
    ):
        self.env = env
        self.state_seq = state_seq

        if isinstance(self.state_seq, list) and len(self.state_seq) > 1:
            self.animate_agents = animate_agents
            self.animate_goals = animate_goals
            self.animate_landmarks = animate_landmarks
        else:
            self.animate_agents = False
            self.animate_goals = False
            self.animate_landmarks = False

        if fps is None:
            self.fps = 8 / env.step_dt
        else:
            self.fps = fps

        if isinstance(self.state_seq, list):
            self.keytimes = [
                round(tmp / len(self.state_seq), 8)
                for tmp in range(len(self.state_seq) - 1)
            ]
            self.keytimes.append(1.0)
            self.keytimes = ";".join(map(str, self.keytimes))
            self.dur = round(len(self.state_seq) / self.fps, 3)
        else:
            self.keytimes = None
            self.dur = None

        self.width = self.env.width
        self.height = self.env.height

        self.landmark_rad = self.env.landmark_rad
        self.agent_rad = self.env.agent_rad
        self.goal_rad = self.env.goal_rad

        if color_step is not None:
            self.color_step = color_step
        else:
            self.color_step = max(360 // self.env.num_agents, 25)

    def render_landmarks(self):
        if (
            isinstance(self.state_seq, list)
            and len(self.state_seq) > 1
            and self.animate_landmarks
        ):
            landmark_pos = {}
            for state in self.state_seq:
                for landmark, (landmark_x, landmark_y) in enumerate(state.landmark_pos):
                    if landmark not in landmark_pos:
                        landmark_pos[landmark] = {"cx": [], "cy": []}
                    landmark_pos[landmark]["cx"].append(float(landmark_x))
                    landmark_pos[landmark]["cy"].append(float(landmark_y))

            state_seq_svg = []
            for i, landmark in enumerate(landmark_pos):
                state_seq_svg.append(f'<circle class="landmark" r="{self.landmark_rad}" fill="#84A1AE">')
                for attribute_name in landmark_pos[landmark]:
                    values = ";".join(
                        map(
                            lambda x: f"{x:.3f}", landmark_pos[landmark][attribute_name]
                        )
                    )
                    state_seq_svg.append(
                        f'<animate attributeName="{attribute_name}" dur="{self.dur}s"'
                    )
                    state_seq_svg.append(
                        f'\tkeyTimes="{self.keytimes}" repeatCount="indefinite"'
                    )
                    state_seq_svg.append(f'\tvalues="{values}"/>')

                state_seq_svg.append("</circle>")
                state_seq_svg.append("\n")

            return "\n".join(state_seq_svg)
        else:
            if isinstance(self.state_seq, list):
                state = self.state_seq[0]
            else:
                state = self.state_seq

            landmark_svg = []
            for landmark_x, landmark_y in state.landmark_pos:
                landmark_x = float(landmark_x)
                landmark_y = float(landmark_y)
                landmark_svg.append(
                    f'<circle class="landmark" cx="{landmark_x:.3f}" cy="{landmark_y:.3f}" r="{self.landmark_rad}" fill="#84A1AE">  </circle>'
                )

            return "\n".join(landmark_svg)

    def render_goals(self):
        if (
            isinstance(self.state_seq, list)
            and len(self.state_seq) > 1
            and self.animate_goals
        ):
            goal_pos = {}
            for state in self.state_seq:
                for goal, (goal_x, goal_y) in enumerate(state.goal_pos):
                    if goal not in goal_pos:
                        goal_pos[goal] = {"cx": [], "cy": []}
                    goal_pos[goal]["cx"].append(float(goal_x))
                    goal_pos[goal]["cy"].append(float(goal_y))

            state_seq_svg = []
            for i, landmark in enumerate(goal_pos):
                color = (i * self.color_step) % 360
                state_seq_svg.append(
                    f'<circle class="goal" r="{self.goal_rad}" fill="hsl({color}, 100%, 50%)">'
                )
                for attribute_name in goal_pos[landmark]:
                    values = ";".join(
                        map(lambda x: f"{x:.3f}", goal_pos[landmark][attribute_name])
                    )
                    state_seq_svg.append(
                        f'<animate attributeName="{attribute_name}" dur="{self.dur}s"'
                    )
                    state_seq_svg.append(
                        f'\tkeyTimes="{self.keytimes}" repeatCount="indefinite"'
                    )
                    state_seq_svg.append(f'\tvalues="{values}"/>')

                state_seq_svg.append("</circle>")
                state_seq_svg.append("\n")

            return "\n".join(state_seq_svg)
        else:
            if isinstance(self.state_seq, list):
                state = self.state_seq[0]
            else:
                state = self.state_seq

            goal_svg = []
            for i, (goal_x, goal_y) in enumerate(state.goal_pos):
                color = (i * self.color_step) % 360
                goal_x = float(goal_x)
                goal_y = float(goal_y)
                goal_svg.append(
                    f'<circle class="goal" cx="{goal_x:.3f}" cy="{goal_y:.3f}" r="{self.goal_rad}" fill="hsl({color}, 100%, 50%)">  </circle>'
                )

            return "\n".join(goal_svg)

    def render_agents(self):
        if (
            isinstance(self.state_seq, list)
            and len(self.state_seq) > 1
            and self.animate_agents
        ):
            agent_pos = {}
            for state in self.state_seq:
                for agent, (agent_x, agent_y) in enumerate(state.agent_pos):
                    if agent not in agent_pos:
                        agent_pos[agent] = {"cx": [], "cy": []}

                    agent_pos[agent]["cx"].append(float(agent_x))
                    agent_pos[agent]["cy"].append(float(agent_y))

            state_seq_svg = []
            for i, agent in enumerate(agent_pos):
                color = (i * self.color_step) % 360
                state_seq_svg.append(
                    f'<circle class="agent" r="{self.agent_rad}" fill="hsl({color}, 70%, 50%)">'
                )
                for attribute_name in agent_pos[agent]:
                    values = ";".join(
                        map(lambda x: f"{x:.3f}", agent_pos[agent][attribute_name])
                    )
                    state_seq_svg.append(
                        f'<animate attributeName="{attribute_name}" dur="{self.dur}s"'
                    )
                    state_seq_svg.append(
                        f'\tkeyTimes="{self.keytimes}" repeatCount="indefinite"'
                    )
                    state_seq_svg.append(f'\tvalues="{values}"/>')

                state_seq_svg.append("</circle>")
                state_seq_svg.append("\n")

            return "\n".join(state_seq_svg)
        else:
            if isinstance(self.state_seq, list):
                state = self.state_seq[0]
            else:
                state = self.state_seq
            agent_svg = []
            for i, (agent_x, agent_y) in enumerate(state.agent_pos):
                color = (i * self.color_step) % 360
                agent_x = float(agent_x)
                agent_y = float(agent_y)
                agent_svg.append(
                    f'<circle class="agent" cx="{agent_x:.3f}" cy="{agent_y:.3f}" r="{self.agent_rad}" fill="hsl({color}, 70%, 50%)">  </circle>'
                )

            return "\n".join(agent_svg)

    def render(self):
        scale = max(self.width, self.height) / 512
        scaled_width = math.ceil(self.width / scale)
        scaled_height = math.ceil(self.height / scale)

        view_box = (-self.width / 2, -self.height / 2, self.width, self.height)

        svg_header = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"',
            f'\twidth="{scaled_width}" height="{scaled_height}" viewBox="{" ".join(map(str, view_box))}">',
        ]
        svg_header = "\n".join(svg_header)

        definitions = [
            "<style>",
            "\t.landmark {{ }}",
            "\t.agent {{ }}",
            f"\t.goal {{stroke: black; stroke-width: {self.goal_rad / 4};}}",
            "</style>",
        ]
        definitions = "\n".join(definitions)

        svg_header = [svg_header, "\n", "<defs>", definitions, "</defs>"]
        svg_header = "\n".join(svg_header)

        svg_landmark = self.render_landmarks()
        svg_goal = self.render_goals()
        svg_agent = self.render_agents()

        return "\n".join(
            [svg_header, "\n", svg_landmark, "\n", svg_goal, "\n", svg_agent, "</svg>"]
        )

    def save_svg(self, filename="test.svg"):
        with open(filename, "w") as svg_file:
            svg_file.write(self.render())

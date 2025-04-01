import math
from typing import Optional

import matplotlib.animation as animation
import matplotlib.pyplot as plt


class Plt_Visualizer(object):
	def __init__(
		self,
		env,
		state_seq: list,
		reward_seq=None,
		path=None,
	):
		self.env = env

		self.interval = 50
		self.state_seq = state_seq
		self.reward_seq = reward_seq
		self.path = path
		
		self.init_render()

	def animate(
		self,
		save_fname: Optional[str] = None,
		view: bool = True,
	):
		"""Anim for 2D fct - x (#steps, #pop, 2) & fitness (#steps, #pop)"""
		ani = animation.FuncAnimation(
			self.fig,
			self.update,
			frames=len(self.state_seq),
			blit=False,
			interval=self.interval,
		)
		# Save the animation to a gif
		if save_fname is not None:
			ani.save(save_fname)

		if view:
			plt.show(block=True)

	def init_render(self):
		from matplotlib.patches import Circle, ConnectionPatch
		state = self.state_seq[0]
		
		self.fig, self.ax = plt.subplots(1, 1, figsize=(self.env.width / 4, self.env.height / 4))
		# self.fig, self.ax = plt.subplots(1, 1, figsize=(5, 5))

		if self.path is not None:
			self.ax.plot(self.path[:, 0], self.path[:, 1], lw=1, alpha=0.7)
		
		xlim = (self.env.width + 1) * self.env.obstacle_size / 2
		ylim = (self.env.height + 1) * self.env.obstacle_size / 2
		self.ax.set_xlim([- xlim, xlim])
		self.ax.set_ylim([- ylim, ylim])
		
		self.agent_artists = []
		self.con_artists = []
		for i in range(self.env.num_agents):
			# agents
			c = Circle(
				state.agent_pos[i], self.env.agent_rad, color="dodgerblue", alpha=0.8,
			)
			self.ax.add_patch(c)
			self.agent_artists.append(c)

			# goals
			c = Circle(
				state.goal_pos[i], self.env.goal_rad, color="blue", alpha=1,
			)
			self.ax.add_patch(c)

			# connections
			con = ConnectionPatch(
				state.agent_pos[i], state.goal_pos[i], "data", "data", lw=0.3, ls="--", color="dimgray", alpha=0.8,
			)
			self.ax.add_patch(con)
			self.con_artists.append(con)

		
		for i in range(self.env.num_landmarks):
			c = Circle(
				state.landmark_pos[i], self.env.landmark_rad, color="black",
			)
			self.ax.add_patch(c)

		self.ax.set_title(f"Step: {state.step}")

		# self.step_counter = self.ax.text(-1.95, 1.95, f"Step: {state.step}", va="top")
			
	def update(self, frame):
		state = self.state_seq[frame]
		for i, c in enumerate(self.agent_artists):
			c.center = state.agent_pos[i]
		for i, con in enumerate(self.con_artists):
			con.xy1 = state.agent_pos[i]
		
		self.ax.set_title(f"Step: {state.step}")

		# self.step_counter.set_text(f"Step: {state.step}")


class SVG_Visualizer:
	def __init__(self, env, state_seq, fps=100, color_step=None):
		self.env = env
		self.state_seq = state_seq

		self.landmark_pos = state_seq[0].landmark_pos
		self.goal_pos = state_seq[0].goal_pos

		self.fps = fps
		
		self.keytimes = [round(tmp / len(self.state_seq), 8) for tmp in range(len(self.state_seq) - 1)]
		self.keytimes.append(1.0)
		self.keytimes = ";".join(map(str, self.keytimes))
		self.dur = round(len(self.state_seq) / fps, 3)

		self.width = self.env.width * self.env.obstacle_size
		self.height = self.env.height * self.env.obstacle_size 

		self.landmark_rad = self.env.landmark_rad
		self.agent_rad = self.env.agent_rad
		self.goal_rad = self.env.goal_rad
		
		if color_step is not None:
			self.color_step = color_step
		else:
			self.color_step = max(360 // self.env.num_agents, 20)

	
	def render_landmark(self):
		landmark_svg = []
		for (landmark_x, landmark_y) in self.landmark_pos:
			landmark_x = float(landmark_x)
			landmark_y = float(landmark_y)
			landmark_svg.append(f'<circle class="landmark" cx="{landmark_x:.3f}" cy="{landmark_y:.3f}">  </circle>')
		
		return '\n'.join(landmark_svg)
	
	def render_goal(self):
		goal_svg = []
		for i, (goal_x, goal_y) in enumerate(self.goal_pos):
			color = (i * self.color_step) % 360
			goal_x = float(goal_x)
			goal_y = float(goal_y)
			goal_svg.append(f'<circle class="goal" cx="{goal_x:.3f}" cy="{goal_y:.3f}" fill="hsl({color}, 100%, 50%)">  </circle>')
		
		return '\n'.join(goal_svg)
	
	def render_agent(self):
		agent_pos = {}
		for state in self.state_seq:
			for agent, (agent_x, agent_y) in enumerate(state.agent_pos):
				if agent not in agent_pos:
					agent_pos[agent] = {'cx': [], 'cy': []}

				agent_pos[agent]['cx'].append(float(agent_x))
				agent_pos[agent]['cy'].append(float(agent_y))
		
		state_seq_svg = []
		for i, agent in enumerate(agent_pos):
			color = (i * self.color_step) % 360
			state_seq_svg.append(f'<circle class="agent" fill="hsl({color}, 70%, 50%)">')
			for attribute_name in agent_pos[agent]:
				values = ";".join(map(lambda x: f'{x:.3f}', agent_pos[agent][attribute_name]))
				state_seq_svg.append(f'<animate attributeName="{attribute_name}" dur="{self.dur}s"')
				state_seq_svg.append(f'\tkeyTimes="{self.keytimes}" repeatCount="indefinite"')
				state_seq_svg.append(f'\tvalues="{values}"/>')
			
			state_seq_svg.append('</circle>')
			state_seq_svg.append('\n')
		
		return '\n'.join(state_seq_svg)
	
	def render(self):
		scale = max(self.width, self.height) / 512
		scaled_width = math.ceil(self.width / scale)
		scaled_height = math.ceil(self.height / scale)

		view_box = (- self.width / 2, - self.height / 2, self.width, self.height)

		svg_header = [
			'<?xml version="1.0" encoding="UTF-8"?>',
			'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"',
			f'\twidth="{scaled_width}" height="{scaled_height}" viewBox="{" ".join(map(str, view_box))}">',
		]
		svg_header = '\n'.join(svg_header)
		
		definitions = [
			'<style>',
			f'\t.landmark {{fill: #84A1AE; r: {self.landmark_rad};}}',
			f'\t.agent {{r: {self.agent_rad};}}',
			f'\t.goal {{stroke: black; stroke-width: {self.goal_rad / 4}; r: {self.goal_rad};}}',
			'</style>',
		]
		definitions = '\n'.join(definitions)

		svg_header = [svg_header, '\n', '<defs>', definitions, '</defs>']
		svg_header = '\n'.join(svg_header)

		svg_landmark = self.render_landmark()
		svg_goal = self.render_goal()
		svg_agent = self.render_agent()

		return '\n'.join([svg_header, '\n', svg_landmark, '\n', svg_goal, '\n', svg_agent, '</svg>'])
	
	def save_svg(self, filename="test.svg"):
		with open(filename, "w") as svg_file:
			svg_file.write(self.render())
			
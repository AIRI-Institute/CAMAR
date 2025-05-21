from typing import Optional

import matplotlib.animation as animation
import matplotlib.pyplot as plt


class Visualizer(object):
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
		
		xlim = (self.env.width + 1) / 2
		ylim = (self.env.height + 1) / 2
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

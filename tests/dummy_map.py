import jax.numpy as jnp

from camar.maps import base_map


class dummy_map(base_map):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

        self.num_agents = kwargs.get("num_agents", 1)
        self.num_landmarks = kwargs.get("num_landmarks", 1)
        self.obstacle_size = kwargs.get("obstacle_size", 0.2)

    @property
    def landmark_rad(self):
        return self.kwargs.get("landmark_rad", 0.1)

    @property
    def agent_rad(self):
        return self.kwargs.get("agent_rad", 0.1)

    @property
    def goal_rad(self):
        return self.kwargs.get("goal_rad", 0.1)

    @property
    def height(self):
        return self.kwargs.get("height", 10.0)

    @property
    def width(self):
        return self.kwargs.get("width", 10.0)

    def reset(self, key):
        landmark_pos = self.kwargs.get("landmark_pos", jnp.array([0.0, 0.0]))
        agent_pos = self.kwargs.get("agent_pos", jnp.array([1.0, 1.0]))
        goal_pos = self.kwargs.get("goal_pos", jnp.array([2.0, 2.0]))
        return (
            key,
            landmark_pos,
            agent_pos,
            goal_pos,
        )

    def update_goals(self, keys, goal_pos, to_update):
        return keys, goal_pos

from typing import Type

import jax.numpy as jnp
from flax import struct
from jax.typing import ArrayLike

from .base import BaseDynamic, PhysicalState
from camar.registry import register_dynamic


@struct.dataclass
class DiffDriveState(PhysicalState):
    agent_pos: ArrayLike  # (num_agents, 2)
    agent_vel: ArrayLike  # (num_agents, 2)
    agent_angle: ArrayLike  # (num_agents, 1)

    @classmethod
    def create(
        cls,
        key: ArrayLike,
        landmark_pos: ArrayLike,
        agent_pos: ArrayLike,
        goal_pos: ArrayLike,
        sizes: "Sizes",  # noqa: F821 see maps/base.py
    ) -> "DiffDriveState":
        num_agents = agent_pos.shape[0]

        # at the start all agents are facing the goal
        # calculate angle (agents x goals)
        ego_goal = goal_pos - agent_pos
        agent_angle = jnp.arctan2(ego_goal[:, 1], ego_goal[:, 0]).reshape(num_agents, 1)
        return cls(
            agent_pos=agent_pos,
            agent_vel=jnp.zeros((num_agents, 2)),
            agent_angle=agent_angle,
        )


@register_dynamic()
class DiffDriveDynamic(BaseDynamic):
    def __init__(
        self,
        linear_speed_max: float = 1.0,
        linear_speed_min: float = -1.0,
        angular_speed_max: float = 2.0,
        angular_speed_min: float = -2.0,
        mass: float = 1.0,
        dt: float = 0.01,
    ):
        self.linear_speed_max = linear_speed_max
        self.linear_speed_min = linear_speed_min
        assert linear_speed_max >= linear_speed_min, "linear_speed_max must be greater than linear_speed_min"

        self.linear_speed_accel = (linear_speed_max - linear_speed_min) / 2  # action space is [-1, 1]

        self.angular_speed_max = angular_speed_max
        self.angular_speed_min = angular_speed_min
        assert angular_speed_max >= angular_speed_min, (
            "angular_speed_max must be greater than angular_speed_min"
        )

        self.angular_speed_accel = (angular_speed_max - angular_speed_min) / 2  # action space is [-1, 1]

        self.mass = mass
        self._dt = dt

    @property
    def action_size(self) -> int:
        return 2

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def state_class(self) -> Type[DiffDriveState]:
        return DiffDriveState

    def integrate(
        self,
        key: ArrayLike,
        force: ArrayLike,
        physical_state: DiffDriveState,
        actions: ArrayLike,
    ) -> DiffDriveState:
        linear_speed = (
            self.linear_speed_accel * (actions[:, 0] + 1) + self.linear_speed_min
        )  # ≈[-1; 1] -> ≈[linear_speed_min; linear_speed_max]
        angular_speed = (
            self.angular_speed_accel * (actions[:, 1] + 1) + self.angular_speed_min
        )  # ≈[-1; 1] -> ≈[angular_speed_min; angular_speed_max]

        linear_speed = jnp.clip(
            linear_speed, min=self.linear_speed_min, max=self.linear_speed_max
        )  # (num_agents, )

        angular_speed = jnp.clip(
            angular_speed, min=self.angular_speed_min, max=self.angular_speed_max
        )  # (num_agents, )

        pos = physical_state.agent_pos  # (num_agents, 2)
        angle = physical_state.agent_angle  # (num_agents, 1)

        vel_x = linear_speed[:, None] * jnp.cos(angle)  # (num_agents, 1)
        vel_y = linear_speed[:, None] * jnp.sin(angle)  # (num_agents, 1)

        vel = jnp.concatenate((vel_x, vel_y), axis=-1)  # (num_agents, 2)
        vel += (force / self.mass) * self.dt

        pos += vel * self.dt  # (num_agents, 2)
        angle += angular_speed[:, None] * self.dt  # (num_agents, 1)

        physical_state = physical_state.replace(
            agent_pos=pos,
            agent_angle=jnp.remainder(angle + jnp.pi, 2 * jnp.pi) - jnp.pi,  # [-pi; pi]
            agent_vel=vel,
        )

        return physical_state

from typing import Type

import jax.numpy as jnp
from flax import struct
from jax.typing import ArrayLike

from .base import BaseDynamic, PhysicalState
from camar.registry import register_dynamic


@struct.dataclass
class HolonomicState(PhysicalState):
    agent_pos: ArrayLike  # (num_agents, 2)
    agent_vel: ArrayLike  # (num_agents, 2)

    @classmethod
    def create(
        cls,
        key: ArrayLike,
        landmark_pos: ArrayLike,
        agent_pos: ArrayLike,
        goal_pos: ArrayLike,
        sizes: "Sizes",  # noqa: F821 see maps/base.py
    ) -> "HolonomicState":
        num_agents = agent_pos.shape[0]
        return cls(
            agent_pos=agent_pos,
            agent_vel=jnp.zeros((num_agents, 2)),
        )


@register_dynamic()
class HolonomicDynamic(BaseDynamic):
    def __init__(
        self,
        accel: float = 5.0,
        max_speed: float = 6.0,
        damping: float = 0.25,
        mass: float = 1.0,
        dt: float = 0.01,
    ):
        self.accel = accel
        assert accel > 0, "acceleration must be positive"

        self.max_speed = max_speed  # negative means no restriction (can lead to integration errors)

        self.damping = damping
        assert 0 <= damping < 1, "damping must be in [0, 1)"

        self.mass = mass
        self._dt = dt

    @property
    def action_size(self) -> int:
        return 2

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def state_class(self) -> Type[HolonomicState]:
        return HolonomicState

    def integrate(
        self,
        key: ArrayLike,
        force: ArrayLike,
        physical_state: HolonomicState,
        actions: ArrayLike,
    ) -> HolonomicState:
        pos = physical_state.agent_pos  # (num_agents, 2)
        vel = physical_state.agent_vel  # (num_agents, 2)

        # semi-implicit euler integration

        vel = (1 - self.damping) * vel
        vel += (force + self.accel * actions) / self.mass * self.dt  # force-based control

        speed = jnp.linalg.norm(vel, axis=-1, keepdims=True)  # (num_agents, 1)
        over_max = vel / speed * self.max_speed  # (num_agents, 2)

        vel = jnp.where((speed > self.max_speed) & (self.max_speed >= 0), over_max, vel)  # (num_agents, 2)

        pos += vel * self.dt

        physical_state = physical_state.replace(
            agent_pos=pos,
            agent_vel=vel,
        )

        return physical_state

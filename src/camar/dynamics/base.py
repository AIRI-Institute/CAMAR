from abc import ABC, abstractmethod
from typing import Type

from flax import struct
from jax.typing import ArrayLike


@struct.dataclass
class PhysicalState(ABC):
    agent_pos: ArrayLike  # (num_agents, 2) # need for collisions and rendering

    @classmethod
    @abstractmethod
    def create(cls, key: ArrayLike, agent_pos: ArrayLike) -> "PhysicalState":
        """
        Create a new physical state from the given key and agent position.
        Used only during env.reset(), agent_pos forwarded from map generator in the environment
        """
        return cls(agent_pos=agent_pos)


class BaseDynamic(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @property
    @abstractmethod
    def action_size(self) -> int:
        pass

    @property
    @abstractmethod
    def dt(self) -> float:
        pass

    @property
    @abstractmethod
    def state_class(self) -> Type[PhysicalState]:
        pass

    @abstractmethod
    def integrate(
        self,
        key: ArrayLike,
        force: ArrayLike,
        physical_state: PhysicalState,
        actions: ArrayLike,
    ) -> PhysicalState:
        """
        Integrate the physical state forward in time.
        You have given the access to:
        - key: jax random key for some stochasticity in the agent dynamics (i.e. noise for actions)
        - force: (num_agents, 2) - collision force (it is possible to ignore it if not needed)
        - current state of the physical system
        - actions: (num_agents, 2) - actions to be applied (from env.step())
        Return the next state of the physical system - recommended to perform only one integration step, as collisions are recalculated every integrate step and frameskip can be done in the environment
        This is the only method that should be jitted and vmapped in the environment.
        """
        pass

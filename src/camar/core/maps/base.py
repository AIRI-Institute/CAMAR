from abc import ABC, abstractmethod
from typing import Tuple

from jax import Array
from jax.typing import ArrayLike


class Map(ABC):

    @property
    @abstractmethod
    def landmark_rad(self) -> float: # TODO: various radiuses
        pass

    @property
    @abstractmethod
    def agent_rad(self) -> float: # TODO: various radiuses
        pass

    @property
    @abstractmethod
    def goal_rad(self) -> float: # TODO: various radiuses
        pass

    @abstractmethod
    def reset(self, key: ArrayLike) -> Tuple[Array, Array, Array]: # Tuple[landmark_pos, agent_pos, goal_pos]
        pass

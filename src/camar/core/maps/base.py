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

    def reset(self, key: ArrayLike) -> Tuple[Array, Array, Array, Array]: # Tuple[PRNGKey_goal, landmark_pos, agent_pos, goal_pos]
        raise NotImplementedError(f"{self.__class__.__name__}.reset is not implemented.")

    def reset_lifelong(self, key: ArrayLike) -> Tuple[Array, Array, Array, Array]:  # Tuple[PRNGKey_goal, landmark_pos, agent_pos, goal_pos]
        raise NotImplementedError(f"{self.__class__.__name__}.reset_lifelong is not implemented. Implement or set lifelong=False.")

    def update_goals(self, keys: ArrayLike, goal_pos: ArrayLike, to_update: ArrayLike) -> Tuple[Array, Array]: # Tuple[PRNGKey_goal, goal_pos]
        raise NotImplementedError(f"{self.__class__.__name__}.update_goals is not implemented. Implement or set lifelong=False.")

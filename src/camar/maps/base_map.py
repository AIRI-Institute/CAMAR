from abc import ABC, abstractmethod
from typing import Tuple

from jax import Array
from jax.typing import ArrayLike

from matplotlib import pyplot as plt


class base_map(ABC):
    @property
    @abstractmethod
    def landmark_rad(self) -> float:
        pass

    @property
    @abstractmethod
    def agent_rad(self) -> float:
        pass

    @property
    @abstractmethod
    def goal_rad(self) -> float:
        pass

    @property
    @abstractmethod
    def height(self) -> float:
        pass

    @property
    @abstractmethod
    def width(self) -> float:
        pass

    def reset(
        self, key: ArrayLike
    ) -> Tuple[
        Array, Array, Array, Array
    ]:  # Tuple[jax.random.key, landmark_pos, agent_pos, goal_pos]
        raise NotImplementedError(
            f"{self.__class__.__name__}.reset is not implemented. Must be implemented if lifelong=False."
        )

    def reset_lifelong(
        self, key: ArrayLike
    ) -> Tuple[
        Array, Array, Array, Array
    ]:  # Tuple[jax.random.key, landmark_pos, agent_pos, goal_pos]
        raise NotImplementedError(
            f"{self.__class__.__name__}.reset_lifelong is not implemented. Must be implemented if lifelong=True."
        )

    def update_goals(
        self, keys: ArrayLike, goal_pos: ArrayLike, to_update: ArrayLike
    ) -> Tuple[Array, Array]:  # Tuple[jax.random.key, goal_pos]
        raise NotImplementedError(
            f"{self.__class__.__name__}.update_goals is not implemented. Must be implemented if lifelong=True."
        )

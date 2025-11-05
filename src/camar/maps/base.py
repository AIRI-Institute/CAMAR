import dataclasses
from abc import ABC, abstractmethod
from typing import Tuple, Type

import jax
from flax import struct
from jax import Array
from jax.typing import ArrayLike


class base_map(ABC):
    def __init__(self):
        self.setup_rad()

    @abstractmethod
    def setup_rad(self):
        """
        Setup radius configuration. Override this method to customize radius handling.

        This method should set up:
        - self.agent_rad: Optional[float] (None for heterogeneous)
        - self.landmark_rad: Optional[float] (None for heterogeneous)
        - self.goal_rad: Optional[float] (None for heterogeneous)
        - self.proportional_goal_rad: Optional[bool] (if goal_rad should be agent_rad / 2.5)
        - Any additional radius-related configuration (i.e. agent_rad_range, etc.)
        """
        # Default implementation - subclasses should override
        self.agent_rad = None
        self.landmark_rad = None
        self.goal_rad = None
        self.proportional_goal_rad = False

        self.agent_rad_range = None
        self.landmark_rad_range = None
        self.goal_rad_range = None

    def generate_sizes(self, key: ArrayLike):
        """
        Generate radius values based on the configuration.

        Args:
            key: JAX random key

        Returns:
            Sizes object with appropriate radius values
        """

        agent_rad_key, landmark_rad_key, goal_rad_key = jax.random.split(key, 3)

        values = {
            "landmark_rad": None,
            "agent_rad": None,
            "goal_rad": None,
        }

        # Generate agent_rad if heterogeneous
        if not self.homogeneous_agents:
            values["agent_rad"] = jax.random.uniform(
                agent_rad_key,
                shape=(self.num_agents,),
                minval=self.agent_rad_range[0],
                maxval=self.agent_rad_range[1],
            )

        # Generate landmark_rad if heterogeneous
        if not self.homogeneous_landmarks:
            values["landmark_rad"] = jax.random.uniform(
                landmark_rad_key,
                shape=(self.num_landmarks,),
                minval=self.landmark_rad_range[0],
                maxval=self.landmark_rad_range[1],
            )

        # Generate goal radii if heterogeneous
        if not self.homogeneous_goals:
            if self.proportional_goal_rad:
                # Use proportional goal radius based on agent radius
                values["goal_rad"] = values["agent_rad"] / 2.5
            else:
                values["goal_rad"] = jax.random.uniform(
                    goal_rad_key,
                    shape=(self.num_agents,),
                    minval=self.goal_rad_range[0],
                    maxval=self.goal_rad_range[1],
                )

        return self.sizes_class.create(**values)

    @property
    def homogeneous_agents(self) -> bool:
        """Whether agents have the same radius. Default is True."""
        return True

    @property
    def homogeneous_landmarks(self) -> bool:
        """Whether landmarks have the same radius. Default is True."""
        return True

    @property
    def homogeneous_goals(self) -> bool:
        """Whether goals have the same radius. Default is True."""
        return True

    @property
    @abstractmethod
    def num_agents(self) -> int:
        """Number of agents in the map."""
        pass

    @property
    @abstractmethod
    def num_landmarks(self) -> int:
        """Number of landmarks in the map."""
        pass

    @property
    @abstractmethod
    def height(self) -> float:
        """Height of the map for rendering."""
        pass

    @property
    @abstractmethod
    def width(self) -> float:
        """Width of the map for rendering."""
        pass

    def _create_sizes_class(self) -> Type["Sizes"]:  # noqa: F821
        fields = []
        if not self.homogeneous_agents:
            fields.append(("agent_rad", ArrayLike))
        if not self.homogeneous_landmarks:
            fields.append(("landmark_rad", ArrayLike))
        if not self.homogeneous_goals:
            fields.append(("goal_rad", ArrayLike))

        # Create a dataclass for storing the sizes of agents, landmarks, and goals
        Sizes = dataclasses.make_dataclass(
            "Sizes",
            fields,
            frozen=False,
            init=False,
        )

        def create(cls, agent_rad: ArrayLike, landmark_rad: ArrayLike, goal_rad: ArrayLike) -> "Sizes":
            values = {}
            if not self.homogeneous_landmarks:
                values["landmark_rad"] = landmark_rad
            if not self.homogeneous_agents:
                values["agent_rad"] = agent_rad
            if not self.homogeneous_goals:
                values["goal_rad"] = goal_rad

            return cls(**values)

        # Create a classmethod for creating an instance of the Sizes dataclass
        Sizes.create = classmethod(create)

        return struct.dataclass(Sizes)

    @property
    def sizes_class(self) -> Type["Sizes"]:  # noqa: F821
        """Class for storing generated sizes of agents, landmarks, and goals dynamically based on the configuration."""
        if not hasattr(self, "_sizes_class"):
            self._sizes_class = self._create_sizes_class()
        return self._sizes_class

    def reset(
        self, key: ArrayLike
    ) -> Tuple[
        Array, Array, Array, Array, "Sizes"  # noqa: F821
    ]:  # Tuple[jax.random.key, landmark_pos, agent_pos, goal_pos, Sizes]
        raise NotImplementedError(
            f"{self.__class__.__name__}.reset is not implemented. Must be implemented if lifelong=False."
        )

    def reset_lifelong(
        self, key: ArrayLike
    ) -> Tuple[Array, Array, Array, Array]:  # Tuple[jax.random.key, landmark_pos, agent_pos, goal_pos]
        raise NotImplementedError(
            f"{self.__class__.__name__}.reset_lifelong is not implemented. Must be implemented if lifelong=True."
        )

    def update_goals(
        self, keys: ArrayLike, goal_pos: ArrayLike, to_update: ArrayLike
    ) -> Tuple[Array, Array]:  # Tuple[jax.random.key, goal_pos]
        raise NotImplementedError(
            f"{self.__class__.__name__}.update_goals is not implemented. Must be implemented if lifelong=True."
        )

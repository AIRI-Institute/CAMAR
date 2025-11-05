import dataclasses
from typing import List, Type

import jax
import jax.numpy as jnp
from flax import struct
from jax.typing import ArrayLike

from camar.dynamics.base import BaseDynamic, PhysicalState
from camar.registry import register_dynamic


@register_dynamic()
class MixedDynamic(BaseDynamic):
    # TODO: dynamics_batch is the list of class instances = impossible to configure using yaml
    def __init__(self, dynamics_batch: List[BaseDynamic], num_agents_batch: List[int]):
        self.dynamics_batch = dynamics_batch
        self.num_agents_batch = num_agents_batch

        assert len(self.dynamics_batch) == len(self.num_agents_batch), (
            "dynamics and num_agents must have the same length"
        )

        dt_0 = self.dynamics_batch[0].dt
        assert all(x.dt == dt_0 for x in self.dynamics_batch), "all dynamics must have the same dt"

        # Create the MixedState dataclass dynamically
        self._create_mixed_state_class()

    def _create_mixed_state_class(self):
        """Create a MixedState dataclass that contains all the individual state classes"""

        # Create fields for the mixed state
        fields = []
        for i, dynamic in enumerate(self.dynamics_batch):
            # Get the state class name and create a field for it
            state_class_name = f"state_{i}"
            fields.append((state_class_name, dynamic.state_class))

        fields.append(("agent_pos", ArrayLike))

        # Create the dataclass
        MixedState = dataclasses.make_dataclass(
            "MixedState",
            fields,
            # bases=(PhysicalState,),  # Inherit from PhysicalState cannot be done because it is frozen
            frozen=False,
            init=False,
        )

        # Add the create method
        def create(
            cls,
            key: ArrayLike,
            landmark_pos: ArrayLike,
            agent_pos: ArrayLike,
            goal_pos: ArrayLike,
            sizes: "Sizes",  # noqa: F821 see maps/base.py
        ) -> "MixedState":
            """Create a mixed state by creating individual states for each dynamic"""
            values = {}

            for i, (dynamic, num_agents) in enumerate(zip(self.dynamics_batch, self.num_agents_batch)):
                # Create individual state for this dynamic
                individual_state = dynamic.state_class.create(
                    key,
                    landmark_pos,
                    agent_pos[i : i + num_agents],
                    goal_pos[i : i + num_agents],
                    jax.tree.map(lambda x: x[i : i + num_agents], sizes),
                )
                values[f"state_{i}"] = individual_state

            values["agent_pos"] = agent_pos  # TODO: this is the copy - may be can be fixed using jax.tree

            # Create the mixed state
            return cls(**values)

        # Add the create method to the class
        MixedState.create = classmethod(create)

        # Apply flax.struct.dataclass decorator
        self._state_class = struct.dataclass(MixedState)

    @property
    def num_agents(self) -> int:
        return sum(self.num_agents_batch)

    @property
    def action_size(self) -> int:
        # actions must be the same size due to vectorization, but it will be sliced see integrate
        return max(map(lambda x: x.action_size, self.dynamics_batch))

    @property
    def dt(self) -> float:
        return self.dynamics_batch[0].dt

    @property
    def state_class(self) -> Type[PhysicalState]:
        return self._state_class

    def integrate(
        self,
        key: ArrayLike,
        force: ArrayLike,
        physical_state: "MixedState",  # noqa: F821 MixedState is defined in _create_mixed_state_class based on dynamics_batch
        actions: ArrayLike,
    ) -> "MixedState":  # noqa: F821
        """Integrate each individual state separately"""
        new_values = {}
        agent_pos = []
        for i, (dynamic, num_agents) in enumerate(zip(self.dynamics_batch, self.num_agents_batch)):
            # Integrate individual state
            new_state = dynamic.integrate(
                key,
                force[i : i + num_agents],
                getattr(physical_state, f"state_{i}"),
                actions[i : i + num_agents, 0 : dynamic.action_size],
            )

            new_values[f"state_{i}"] = new_state
            agent_pos.append(new_state.agent_pos)

        new_values["agent_pos"] = jnp.concatenate(agent_pos, axis=0)

        # Return the updated mixed state
        return physical_state.replace(**new_values)

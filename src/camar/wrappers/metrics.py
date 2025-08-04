# From Craftax Baselines https://github.com/MichaelTMatthews/Craftax_Baselines.git with added a success rate metric

from functools import partial
from typing import Any, Union

import jax
from flax import struct
from jax.typing import ArrayLike

from camar.wrappers import GymnaxWrapper


@struct.dataclass
class LogEnvState:
    env_state: Any
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    timestep: int
    success_rate: float


class LogWrapper(GymnaxWrapper):
    """Log the episoreturns and lengths."""

    def __init__(self, env):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: ArrayLike):
        obs, env_state = self._env.reset(key)
        state = LogEnvState(env_state, 0.0, 0, 0.0, 0, 0)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: ArrayLike,
        state,
        action: Union[int, float],
    ):
        obs, env_state, reward, done, info = self._env.step(key, state.env_state, action)
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done) + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done) + new_episode_length * done,
            timestep=state.timestep + 1,
            success_rate=env_state.on_goal.sum(axis=-1) / self.num_agents,
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["timestep"] = state.timestep
        info["returned_episode"] = done

        info["success_rate"] = state.success_rate

        return obs, state, reward, done, info

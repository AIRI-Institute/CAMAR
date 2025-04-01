
import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data.tensor_specs import Bounded, Categorical, Composite, Unbounded
from torchrl.envs.common import _EnvWrapper
from torchrl.envs.libs.jax_utils import (
    _extract_spec,
    _ndarray_to_tensor,
    _object_to_tensordict,
    _tensor_to_ndarray,
    _tree_flatten,
)
from torchrl.envs.utils import MarlGroupMapType, _classproperty, check_marl_grouping


class MyEnvWrapper(_EnvWrapper):
    _jax = None

    @_classproperty
    def jax(cls):
        if cls._jax is not None:
            return cls._jax

        import jax

        cls._jax = jax
        return jax

    def __init__(self, env=None, **kwargs):
        if env is not None:
            kwargs["env"] = env

        super().__init__(**kwargs)

    def _check_kwargs(self, kwargs: dict):
        if "env" not in kwargs:
            raise TypeError("Could not find environment key 'env' in kwargs.")

    def _build_env(
        self,
        env,
    ):
        return env

    def _make_state_spec(self, env):  # noqa: F821
        jax = self.jax

        key = jax.random.PRNGKey(0)
        state = env.reset(key)
        state_dict = _object_to_tensordict(state, self.device, batch_size=())
        state_spec = _extract_spec(state_dict).expand(self.batch_size)
        return state_spec

    def _make_specs(self, env) -> None:  # noqa: F821
        agent_names = [f"agent_{agent_idx}" for agent_idx in range(env.num_agents)]
        self.group_map = MarlGroupMapType.ALL_IN_ONE_GROUP.get_group_map(agent_names)

        # just to be confident
        check_marl_grouping(self.group_map, agent_names)
        assert len(self.group_map.keys()) == 1
        assert "agents" in self.group_map.keys()

        action = Bounded(
            low=-1,
            high=1,
            shape=(*self.batch_size, env.num_agents, env.action_size),
            device=self.device,
        )
        agents_action = Composite(
            action=action,
            shape=(*self.batch_size, env.num_agents),
            device=self.device,
        )
        self.action_spec = Composite(
            agents=agents_action,
            shape=self.batch_size,
            device=self.device,
        )

        reward = Bounded(
            low=-0.5,
            high=1,
            shape=(*self.batch_size, env.num_agents, 1),
            device=self.device,
        )
        agents_reward = Composite(
            reward=reward,
            shape=(*self.batch_size, env.num_agents),
            device=self.device,
        )
        self.reward_spec = Composite(
            agents=agents_reward,
            shape=self.batch_size,
            device=self.device,
        )

        observation = Unbounded(
            shape=(*self.batch_size, env.num_agents, env.observation_size),
            device=self.device,
        )
        agents_observation = Composite(
            observation=observation,
            shape=(*self.batch_size, env.num_agents),
            device=self.device,
        )
        self.observation_spec = Composite(
            agents=agents_observation,
            shape=self.batch_size,
            device=self.device,
        )

        self.done_spec = Categorical(
            n=2,
            shape=(*self.batch_size, 1),
            dtype=torch.bool,
            device=self.device,
        )

    # def _make_state_example(self):
    #     jax = self.jax

    #     key = jax.random.PRNGKey(0)
    #     keys = jax.random.split(key, self.batch_size.numel())
    #     state, obs, done = self._vmap_jit_env_reset(jax.numpy.stack(keys))
    #     # state = _tree_reshape(state, self.batch_size)
    #     return state

    def _init_env(self) -> int | None:
        jax = self.jax
        self._key = None
        self._vmap_jit_env_reset = jax.vmap(self._env.reset)
        self._vmap_jit_env_step = jax.vmap(self._env.step)
        # self._state_example = self._make_state_example()

    def _set_seed(self, seed: int):
        jax = self.jax
        if seed is None:
            raise Exception("Brax requires an integer seed.")
        self._key = jax.random.PRNGKey(seed)

    def _reset(self, tensordict: TensorDictBase = None, **kwargs) -> TensorDictBase:
        jax = self.jax

        # generate random keys
        self._key, *keys = jax.random.split(self._key, 1 + self.numel())

        # call env reset with jit and vmap
        self._state, obs, done = self._vmap_jit_env_reset(jax.numpy.stack(keys))

        tensordict_agents = TensorDict(
            source={
                "observation": _ndarray_to_tensor(obs),
            },
            batch_size=(*self.batch_size, self._env.num_agents),
            device=self.device,
        )

        done = _ndarray_to_tensor(done)

        tensordict_out = TensorDict(
            source={
                "agents": tensordict_agents,
                "done": done,
                "terminated": done.clone(),
            },
            batch_size=self.batch_size,
            device=self.device,
        )
        return tensordict_out

    def _step(self, tensordict: TensorDictBase):
        jax = self.jax

        # convert tensors to ndarrays
        # state = _tensordict_to_object(tensordict.get("state"), self._state_example)

        action = _tensor_to_ndarray(tensordict.get(("agents", "action")))

        # flatten batch size
        # state = _tree_flatten(state, self.batch_size)
        action = _tree_flatten(action, self.batch_size)

        # call env step with jit and vmap
        self._key, *keys_s = jax.random.split(self._key, 1 + self.numel())

        self._state, obs, reward, done = self._vmap_jit_env_step(
            jax.numpy.stack(keys_s), self._state, action
        )

        tensordict_agents = TensorDict(
            source={
                "observation": _ndarray_to_tensor(obs),
                "reward": _ndarray_to_tensor(reward),
            },
            batch_size=(*self.batch_size, self._env.num_agents),
            device=self.device,
        )

        done = _ndarray_to_tensor(done)

        tensordict_out = TensorDict(
            source={
                "agents": tensordict_agents,
                "done": done,
                "terminated": done.clone(),
            },
            batch_size=self.batch_size,
            device=self.device,
        )
        return tensordict_out

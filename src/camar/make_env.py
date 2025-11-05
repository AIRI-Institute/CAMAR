from typing import Optional, Union

from camar.dynamics import BaseDynamic

from .environment import Camar
from .maps import base_map
from .registry import DYNAMIC_REGISTRY, MAP_REGISTRY


def _instantiate_from_factory(factory, kwargs: Optional[dict]):
    return factory(**(kwargs or {}))


def _resolve_map(map_spec, map_kwargs):
    # get instance
    if isinstance(map_spec, base_map):
        return map_spec

    # get registered name
    if isinstance(map_spec, str) and map_spec in MAP_REGISTRY:
        return _instantiate_from_factory(MAP_REGISTRY[map_spec], map_kwargs)

    # get subclass of base_map
    if isinstance(map_spec, type) and issubclass(map_spec, base_map):
        return _instantiate_from_factory(map_spec, map_kwargs)

    raise TypeError("map_generator must be an instance, subclass, or registered name of camar.maps.base_map")


def _resolve_dynamic(dyn_spec, dyn_kwargs):
    # get instance
    if isinstance(dyn_spec, BaseDynamic):
        return dyn_spec

    # get registered name
    if isinstance(dyn_spec, str) and dyn_spec in DYNAMIC_REGISTRY:
        return _instantiate_from_factory(DYNAMIC_REGISTRY[dyn_spec], dyn_kwargs)

    # get subclass of BaseDynamic
    if isinstance(dyn_spec, type) and issubclass(dyn_spec, BaseDynamic):
        return _instantiate_from_factory(dyn_spec, dyn_kwargs)

    raise TypeError("dynamic must be an instance, subclass, or registered name of camar.dynamics.BaseDynamic")


def make_env(
    map_generator: Optional[Union[str, base_map, type]] = "random_grid",
    dynamic: Optional[Union[str, BaseDynamic, type]] = "HolonomicDynamic",
    lifelong: bool = False,
    window: float = 0.3,
    max_steps: int = 100,
    frameskip: int = 2,
    max_obs: int = 3,
    pos_shaping_factor: float = 1.0,
    contact_force: float = 500,
    contact_margin: float = 0.001,
    map_kwargs: Optional[dict] = None,
    dynamic_kwargs: Optional[dict] = None,
):
    map_obj = _resolve_map(map_generator, map_kwargs)
    dyn_obj = _resolve_dynamic(dynamic, dynamic_kwargs)

    env = Camar(
        map_generator=map_obj,
        dynamic=dyn_obj,
        lifelong=lifelong,
        window=window,
        max_steps=max_steps,
        frameskip=frameskip,
        max_obs=max_obs,
        pos_shaping_factor=pos_shaping_factor,
        contact_force=contact_force,
        contact_margin=contact_margin,
    )

    return env

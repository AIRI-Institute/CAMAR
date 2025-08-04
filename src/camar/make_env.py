import importlib
from typing import Optional, Union

from camar.dynamics import BaseDynamic

from .environment import Camar
from .maps import base_map

MAPS_MODULE = "camar.maps"
DYNAMICS_MODULE = "camar.dynamics"


def make_env(
    map_generator: Optional[Union[str, base_map]] = "random_grid",
    dynamic: Optional[Union[str, BaseDynamic]] = "HolonomicDynamic",
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
    if isinstance(map_generator, str):
        module = importlib.import_module(MAPS_MODULE)
        if map_kwargs is not None:
            map_generator = getattr(module, map_generator)(**map_kwargs)
        else:
            map_generator = getattr(module, map_generator)()

    if isinstance(dynamic, str):
        module = importlib.import_module(DYNAMICS_MODULE)
        if dynamic_kwargs is not None:
            dynamic = getattr(module, dynamic)(**dynamic_kwargs)
        else:
            dynamic = getattr(module, dynamic)()

    env = Camar(
        map_generator=map_generator,
        dynamic=dynamic,
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

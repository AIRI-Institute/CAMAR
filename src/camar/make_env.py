import importlib
from typing import Optional, Union

from .environment import Camar
from .maps import random_grid
from .maps import base_map

MAPS_MODULE = "camar.maps"


def make_env(
    map_generator: Optional[Union[str, base_map]] = random_grid(),
    window: float = 0.8,
    placeholder: float = 0.0,
    max_steps: int = 100,
    frameskip: int = 2,
    max_obs: Optional[int] = None,
    dt: float = 0.01,
    damping: float = 0.25,
    contact_force: float = 500,
    contact_margin: float = 0.001,
    **map_kwargs,
):
    if isinstance(map_generator, str):
        module = importlib.import_module(MAPS_MODULE)
        map_generator = getattr(module, map_generator)(**map_kwargs)

    env = Camar(
        map_generator=map_generator,
        window=window,
        placeholder=placeholder,
        max_steps=max_steps,
        frameskip=frameskip,
        max_obs=max_obs,
        dt=dt,
        damping=damping,
        contact_force=contact_force,
        contact_margin=contact_margin,
    )

    return env

from functools import partial
from typing import List

import jax
import jax.numpy as jnp

from .base_map import base_map
from .batched_string_grid import batched_string_grid
from .const import CPU_DEVICE
from .utils import detect_edges, get_movingai


def preprocess(map_array, height, width, low_thr, return_edges=True):
    import cv2
    import numpy as np

    map_array = np.array(map_array)
    map_array = cv2.resize(map_array, (height, width), interpolation=cv2.INTER_NEAREST)
    map_array = jnp.asarray(map_array, device=CPU_DEVICE)

    if return_edges:
        map_array = detect_edges(map_array, low_thr)
    return map_array


class movingai(batched_string_grid):
    def __init__(
        self,
        map_names: List[str],
        height: int = 128,
        width: int = 128,
        low_thr: float = 3.7,
        only_edges: bool = True,
        remove_border: bool = True,
        add_border: bool = False,
        num_agents: int = 10,
        obstacle_size: float = 0.1,
        agent_size: float = 0.06,
    ) -> base_map:
        map_str_batch = get_movingai(map_names)

        map_array_preprocess = partial(
            preprocess,
            height=height,
            width=width,
            low_thr=low_thr,
            return_edges=only_edges,
        )
        free_pos_array_preprocess = partial(
            preprocess,
            height=height,
            width=width,
            low_thr=low_thr,
            return_edges=False,
        )

        super().__init__(
            map_str_batch=map_str_batch,
            free_pos_str_batch=map_str_batch,
            agent_idx_batch=None,
            goal_idx_batch=None,
            num_agents=num_agents,
            random_agents=True,
            random_goals=True,
            remove_border=remove_border,
            add_border=add_border,
            obstacle_size=obstacle_size,
            agent_size=agent_size,
            map_array_preprocess=map_array_preprocess,
            free_pos_array_preprocess=free_pos_array_preprocess,
        )

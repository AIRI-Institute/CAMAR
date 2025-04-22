from functools import partial
from typing import List

import jax
import jax.numpy as jnp

from .base_map import base_map
from .batched_string_grid import batched_string_grid

cpu_device = jax.devices("cpu")[0]


def get_movingai(map_names):
    import io
    import os
    import zipfile

    import requests

    movingai_path = ".cache/movingai/"
    if not os.path.exists(movingai_path):
        os.makedirs(movingai_path)

    zip_files = {}
    map_str_batch = []
    for collection, map_name in map(lambda x: x.split("/"), map_names):
        path_to_map = f".cache/movingai/{collection}/{map_name}.map"
        if not os.path.exists(path_to_map):
            if collection not in zip_files:
                url_collection = (
                    f"https://movingai.com/benchmarks/{collection}/{collection}-map.zip"
                )
                response = requests.get(url_collection)
                zip_file = io.BytesIO(response.content)

                z = zipfile.ZipFile(zip_file, "r")
                zip_files[collection] = z

            z = zip_files[collection]
            map_file_name = f"{map_name}.map"
            if map_file_name not in z.namelist():
                raise ValueError(f"there is no {map_file_name} in {url_collection=}.")

            with z.open(map_file_name, "r") as f:
                map_str = f.read().decode()
                map_str = map_str.split("\n")[4:]
                map_str_batch.append("\n".join(map_str))
                os.makedirs(os.path.dirname(path_to_map), exist_ok=True)
                with open(path_to_map, "w") as output_f:
                    output_f.write(map_str)
        else:
            with open(path_to_map, "r") as f:
                map_str = f.read()
                map_str = map_str.split("\n")[4:]
                map_str_batch.append("\n".join(map_str))

    for collection, z in zip_files.items():
        z.close()

    return map_str_batch


def get_edges(img, low_thr):
    sobel_x = jnp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=cpu_device)

    sobel_y = jnp.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=cpu_device)

    edges_x = jax.scipy.signal.convolve2d(img, sobel_x)
    edges_y = jax.scipy.signal.convolve2d(img, sobel_y)

    edges = jnp.sqrt(edges_x**2 + edges_y**2)

    return edges > low_thr


def preprocess(map_array, height, width, low_thr):
    import cv2
    import numpy as np

    map_array = np.array(map_array)
    map_array = cv2.resize(map_array, (height, width), interpolation=cv2.INTER_NEAREST)
    map_array = jnp.asarray(map_array, device=cpu_device)
    map_array = get_edges(map_array, low_thr)
    return map_array


class movingai(batched_string_grid):
    def __init__(
        self,
        map_names: List[str],
        height: int = 128,
        width: int = 128,
        low_thr: float = 3.7,
        num_agents: int = 10,
        obstacle_size: float = 0.1,
        agent_size: float = 0.06,
    ) -> base_map:
        map_str_batch = get_movingai(map_names)

        super().__init__(
            map_str_batch=map_str_batch,
            agent_idx_batch=None,
            goal_idx_batch=None,
            num_agents=num_agents,
            random_agents=True,
            random_goals=True,
            remove_border=False,
            add_border=True,
            obstacle_size=obstacle_size,
            agent_size=agent_size,
            map_array_preprocess=partial(
                preprocess, height=height, width=width, low_thr=low_thr
            ),
        )

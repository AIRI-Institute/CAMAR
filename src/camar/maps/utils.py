import jax
import jax.numpy as jnp

from .const import PREGEN_DEVICE


def map_str2array(map_str, remove_border, add_border, preprocess=lambda map_array: map_array):
    map_array = []
    for line in map_str.split("\n"):
        line_array = []
        if set(line) == set(" ") or not line:
            continue
        for char in line.strip():
            if char in set("@*#"):
                line_array.append(1)
            else:
                line_array.append(0)
        map_array.append(line_array)

    map_array = jnp.array(map_array, device=PREGEN_DEVICE)

    map_array = preprocess(map_array)

    if remove_border:
        map_array = map_array[1:-1, 1:-1]

    if add_border:
        map_array = jnp.pad(map_array, pad_width=[(1, 1), (1, 1)], constant_values=[(1, 1), (1, 1)])

    return map_array


def idx2pos(idx_x, idx_y, obstacle_size, height, width):
    coord_x = idx_x * obstacle_size - height / 2 + obstacle_size / 2
    coord_y = idx_y * obstacle_size - width / 2 + obstacle_size / 2

    return jnp.stack((coord_y, coord_x), axis=1)


def parse_map_array(map_array, obstacle_size, free_pos_array=None):
    num_rows, num_cols = map_array.shape

    map_idx_rows, map_idx_cols = jnp.meshgrid(
        jnp.arange(num_cols, device=PREGEN_DEVICE),
        jnp.arange(num_rows, device=PREGEN_DEVICE),
    )

    height = num_rows * obstacle_size
    width = num_cols * obstacle_size

    # obstacles
    landmark_idx_x, landmark_idx_y = jnp.nonzero(map_array)
    landmark_pos = idx2pos(landmark_idx_x, landmark_idx_y, obstacle_size, height, width)

    # free cells
    map_idx = jnp.stack((map_idx_cols, map_idx_rows), axis=2).reshape(
        -1, 2
    )  # for random agent and goal positions
    if free_pos_array is None:
        is_free = ~map_array.flatten().astype(jnp.bool_)
    else:
        map_array_free = ~map_array.flatten().astype(jnp.bool_)
        is_free = ~free_pos_array.flatten().astype(jnp.bool_)
        is_free = jnp.logical_and(map_array_free, is_free)

    free_idx = map_idx[is_free, :]
    free_pos = idx2pos(free_idx[:, 0], free_idx[:, 1], obstacle_size, height, width)

    return landmark_pos, free_pos, height, width


# for batched string grid
def pad_placeholder(pos, num_pos, placeholder=-100.0):
    return jnp.concatenate(
        (
            pos,
            jnp.full(
                shape=(num_pos - pos.shape[0], pos.shape[1]),
                fill_value=placeholder,
                dtype=pos.dtype,
                device=PREGEN_DEVICE,
            ),
        ),
        axis=0,
    )


def random_truncate(pos, num_pos, key=jax.random.key(0)):
    return jax.random.choice(key, pos, shape=(num_pos,), replace=False)


def check_pos(map_array, pos):
    agent_cells = map_array[pos[:, 0], pos[:, 1]]
    return ~agent_cells.any()


# random_grid
def get_border_landmarks(num_rows, num_cols, half_width, half_height, grain_factor):
    top_wall = jnp.stack(
        (
            jnp.linspace(
                -half_width,  # start
                half_width,  # end
                num_rows * (grain_factor - 1),  # num points
                endpoint=False,
            ),
            jnp.full(
                (num_rows * (grain_factor - 1),),  # num points
                half_height,
            ),  # y coord of the top wall
        ),
        axis=-1,
    )
    right_wall = jnp.stack(
        (
            jnp.full(
                (num_cols * (grain_factor - 1),),  # num points
                half_width,
            ),  # x coord of the right wall
            jnp.linspace(
                half_height,  # start
                -half_height,  # end
                num_cols * (grain_factor - 1),  # num points
                endpoint=False,
            ),
        ),
        axis=-1,
    )
    bottom_wall = jnp.stack(
        (
            jnp.linspace(
                half_width,  # start
                -half_width,  # end
                num_rows * (grain_factor - 1),  # num points
                endpoint=False,
            ),
            jnp.full(
                (num_rows * (grain_factor - 1),),  # num points
                -half_height,
            ),  # y coord of the bottom wall
        ),
        axis=-1,
    )
    left_wall = jnp.stack(
        (
            jnp.full(
                (num_cols * (grain_factor - 1),),  # num points
                -half_width,
            ),  # x coord of the left wall
            jnp.linspace(
                -half_height,  # start
                half_height,  # end
                num_cols * (grain_factor - 1),  # num points
                endpoint=False,
            ),
        ),
        axis=-1,
    )
    return jnp.concatenate([top_wall, right_wall, left_wall, bottom_wall])


# movingai
def delete_movingai_header(map_str):
    map_str = map_str.replace("\r", "").split("\n")[4:]
    map_str = "\n".join(map_str)
    return map_str


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
                url_collection = f"https://movingai.com/benchmarks/{collection}/{collection}-map.zip"
                response = requests.get(url_collection)
                zip_file = io.BytesIO(response.content)

                z = zipfile.ZipFile(zip_file, "r")
                zip_files[collection] = z

            z = zip_files[collection]
            map_file_name = f"{map_name}.map"
            if map_file_name not in z.namelist():
                raise ValueError(f"there is no {map_file_name} in {url_collection}.")

            with z.open(map_file_name, "r") as f:
                map_str = f.read().decode()
                map_str_batch.append(delete_movingai_header(map_str))

                os.makedirs(os.path.dirname(path_to_map), exist_ok=True)
                with open(path_to_map, "w") as output_f:
                    output_f.write(map_str)
        else:
            with open(path_to_map, "r") as f:
                map_str = f.read()
                map_str_batch.append(delete_movingai_header(map_str))

    for collection, z in zip_files.items():
        z.close()

    return map_str_batch


def detect_edges(img, low_thr, mode="same"):
    sobel_x = jnp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=PREGEN_DEVICE)

    sobel_y = jnp.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=PREGEN_DEVICE)

    edges_x = jax.scipy.signal.convolve2d(img, sobel_x, mode=mode)
    edges_y = jax.scipy.signal.convolve2d(img, sobel_y, mode=mode)

    edges = jnp.sqrt(edges_x**2 + edges_y**2)

    return edges > low_thr


def fade(t):
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def lerp(a, b, t):
    return a + t * (b - a)


def generate_gradients(key, grid_width, grid_height):
    angles = jax.random.uniform(key, shape=(grid_width + 1, grid_height + 1), minval=0.0, maxval=2.0 * jnp.pi)
    gradients = jnp.dstack((jnp.cos(angles), jnp.sin(angles)))
    return gradients


def perlin_noise_vectorized(key, width, height, grid_width, grid_height):
    gradients = generate_gradients(key, grid_width, grid_height)

    xs = jnp.linspace(0, grid_width, width, endpoint=False)
    ys = jnp.linspace(0, grid_height, height, endpoint=False)
    x, y = jnp.meshgrid(xs, ys)

    x0 = x.astype(int)
    y0 = y.astype(int)
    x1 = x0 + 1
    y1 = y0 + 1

    sx = fade(x - x0)
    sy = fade(y - y0)

    # Top-left
    n00 = (x - x0) * gradients[x0, y0, 0] + (y - y0) * gradients[x0, y0, 1]
    # Top-right
    n10 = (x - x1) * gradients[x1, y0, 0] + (y - y0) * gradients[x1, y0, 1]
    # Bottom-left
    n01 = (x - x0) * gradients[x0, y1, 0] + (y - y1) * gradients[x0, y1, 1]
    # Bottom-right
    n11 = (x - x1) * gradients[x1, y1, 0] + (y - y1) * gradients[x1, y1, 1]

    # Interpolate the dot products along the x-direction.
    nx0 = lerp(n00, n10, sx)
    nx1 = lerp(n01, n11, sx)

    # Interpolate the results along the y-direction to get the final noise.
    nxy = lerp(nx0, nx1, sy)

    return nxy

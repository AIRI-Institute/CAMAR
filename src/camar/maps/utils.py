import jax
import jax.numpy as jnp

cpu_device = jax.devices("cpu")[0]


def map_str2array(map_str, remove_border, add_border, preprocess=lambda map_array: map_array):
    map_array = jnp.array(
        [
            [1 if char in set("@*#") else 0 for char in line]
            for line in map_str.split("\n")
            if line
        ],
        device=cpu_device,
    )

    map_array = preprocess(map_array)

    if remove_border:
        map_array = map_array[1:-1, 1:-1]

    if add_border:
        map_array = jnp.pad(
            map_array, pad_width=[(1, 1), (1, 1)], constant_values=[(1, 1), (1, 1)]
        )

    return map_array


def idx2pos(idx_x, idx_y, obstacle_size, height, width):
    coord_x = idx_x * obstacle_size - height / 2 + obstacle_size / 2
    coord_y = idx_y * obstacle_size - width / 2 + obstacle_size / 2

    return jnp.stack((coord_y, coord_x), axis=1)


def parse_map_array(map_array, obstacle_size):
    num_rows, num_cols = map_array.shape

    map_idx_rows, map_idx_cols = jnp.meshgrid(
        jnp.arange(num_cols, device=cpu_device), jnp.arange(num_rows, device=cpu_device)
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
    is_free = ~map_array.flatten().astype(jnp.bool_)
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
                device=cpu_device,
            ),
        ),
        axis=0,
    )


def random_truncate(pos, num_pos, key=jax.random.key(0)):
    return jax.random.choice(key, pos, shape=(num_pos,), replace=False)


def check_pos(map_array, pos):
    agent_cells = map_array[pos[:, 0], pos[:, 1]]
    return ~agent_cells.any()

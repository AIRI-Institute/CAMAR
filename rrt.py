from functools import partial

import chex
import jax
import jax.numpy as jnp
from flax import struct


@partial(jax.vmap, in_axes=[0, 0, None, None])
@partial(jax.jit, static_argnames=("effective_rad",))
def check_collision(start_point: chex.Array, end_point: chex.Array, obstacles: chex.Array, effective_rad: float):
    """
    Calculate whether there is a collision between a line segment
    defined by start and end points and a set of circle obstacles.

    :param start_point: The `start_point` parameter represents the starting point of the agent's path.

    :param end_point: The `end_point` parameter represents the end point of the agent's path.

    :param obstacles: The positions of obstacles in the environment with shape
    `(num_obstacles, 2)`.

    :param effective_rad: The effective radius within which a collision is considered to have
    occurred between the agent's path and the obstacles.

    :return: A boolean value indicating whether there is a collision between the agent's path
    and the obstacles.
    """
    # Compute the vector along the agent's path.
    start_to_end = end_point - start_point  # (2, )
    start_to_end_length_sq = jnp.sum(start_to_end * start_to_end)  # ()

    # Compute vectors from start_point to each obstacle center.
    start_to_obs = obstacles - start_point[None, :]  # (num_obstacles, 2)

    # Compute the projection scalar (t) for each obstacle onto the line SE.
    # This gives the position along SE where the perpendicular from the obstacle falls.
    t = (
        jnp.sum(start_to_obs * start_to_end[None, :], axis=1) / start_to_end_length_sq
    )  # (num_obstacles, 2)

    # Clamp t to the interval [0, 1] so that we only consider the segment.
    t = jnp.clip(t, 0.0, 1.0)

    # Compute the closest point on the segment for each obstacle.
    # Broadcasting t to scale the vector SE for each obstacle.
    closest_points = (
        start_point[None, :] + t[:, None] * start_to_end[None, :]
    )  # (num_obstacles, 2)

    # Compute the squared distances from each obstacle center to its closest point on the segment.
    dist = jnp.linalg.norm(closest_points - obstacles, axis=1)

    return jnp.any(dist <= effective_rad)


@struct.dataclass
class RRTState:
    key: chex.PRNGKey
    pos: chex.Array  # (num_samples, num_agents, [x, y])
    parent: chex.Array  # (num_samples, num_agents)
    goal_reached: chex.Array  # (num_agents, )
    idx: int


class RRT:
    def __init__(
        self, env, num_samples, step_size, collision_func=check_collision
    ):
        self.num_samples = num_samples
        self.step_size = step_size

        self.agent_rad = env.agent_rad
        self.num_agents = env.num_agents
        self.goal_rad = env.goal_rad
        self.effective_rad = env.agent_rad + env.landmark_rad
        self.sample_limit = jnp.array(
            [env.width / 2, env.height / 2]
        )

        self.check_collision = collision_func

    @partial(jax.jit, static_argnames=("self",))
    def run(
        self,
        key: chex.PRNGKey,
        start: chex.Array,
        goal: chex.Array,
        landmark_pos: chex.Array,
    ):
        """
        Run RRT algorithm to find a path between start and goal
        start (num_agents, 2)
        goal (num_agents, 2)
        landmark_pos (num_landmarks, 2)
        """

        @jax.jit
        def _condition(state: RRTState):
            return (state.idx < self.num_samples) & ~jnp.all(state.goal_reached)

        @jax.jit
        def _step(state: RRTState):
            """
            :param state: The `state` parameter in the `_step` function represents the current state of the
            RRT.
            :type state: RRTState
            :return: updated `state` variable after performing the RRT step.
            """
            new_key, key_s = jax.random.split(state.key)

            # for agents reached their goals (if previous on goal then stay on it)
            gr_pos = state.pos[state.idx - 1, :, :]  # (num_agents, 2)

            # Sample position, find closest idx and increment towards sampled pos
            sampled_pos = jax.random.uniform(
                key_s,
                (self.num_agents, 2),
                minval=-self.sample_limit,
                maxval=self.sample_limit,
            )

            distances = jnp.linalg.norm(
                state.pos[:, :, :] - sampled_pos[None, :, :], axis=-1
            )  # (rrt_samples, num_agents)
            # closest with previous (valid) samples
            distances = jnp.where(
                (jnp.arange(self.num_samples) < state.idx)[:, None], distances, jnp.inf
            )
            distances = jnp.where(state.parent == -2.0, jnp.inf, distances)
            closest_idx = jnp.argmin(distances, axis=0)  # (num_agents, )

            tree_pos = state.pos[
                closest_idx, jnp.arange(self.num_agents), :
            ]  # (num_agents, 2)

            direction = sampled_pos - tree_pos
            direction_dist = distances[
                closest_idx, jnp.arange(self.num_agents)
            ]  # (num_agents, )
            clipped_direction = jnp.where(
                (direction_dist > self.step_size)[:, None],
                direction / direction_dist[:, None] * self.step_size,
                direction,
            )  # (num_agents, 2)
            test_pos = tree_pos + clipped_direction

            # Check free space, line collision
            valid = ~self.check_collision(
                tree_pos, test_pos, landmark_pos, self.effective_rad
            )  # (num_agents, )

            new_pos = jnp.where(valid[:, None], test_pos, -2.0)
            new_parent = jnp.where(valid, closest_idx, -2)

            new_pos = jnp.where(state.goal_reached[:, None], gr_pos, new_pos)
            new_parent = jnp.where(state.goal_reached, -1, new_parent)

            gr_test = jnp.linalg.norm(test_pos - goal, axis=-1) < self.goal_rad
            gr_test = jnp.where(valid, gr_test, False)
            goal_reached = jnp.where(state.goal_reached, True, gr_test)

            state = state.replace(
                key=new_key,
                pos=state.pos.at[state.idx].set(new_pos),
                parent=state.parent.at[state.idx].set(new_parent),
                goal_reached=goal_reached,
                idx=state.idx + 1 * jnp.any(valid),
            )

            return state

        state = RRTState(
            key=key,
            pos=jnp.full(
                (self.num_samples, self.num_agents, 2), start, dtype=jnp.float32
            ),
            parent=jnp.full((self.num_samples, self.num_agents), -1, dtype=jnp.int32),
            goal_reached=jnp.linalg.norm(start - goal, axis=-1) < self.goal_rad,
            idx=1,
        )

        res_state = jax.lax.while_loop(_condition, _step, state)
        return res_state

    @partial(jax.jit, static_argnames=["self"])
    def find_last_idx(self, state: RRTState):
        idx = jnp.arange(self.num_samples)
        valid_idx = jnp.where(
            (state.parent != -1) & (state.parent != -2), idx[:, None], -1
        )
        return jnp.max(valid_idx, axis=0)

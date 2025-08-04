from functools import partial

import jax
import jax.numpy as jnp
from flax import struct
from jax import Array

from rrt import RRTState


@partial(jax.vmap, in_axes=[0, 0, None, None])
def check_collision(start_point: Array, end_point: Array, obstacles: Array, effective_rad: float):
    # Compute the vector along the agent's path.
    start_to_end = end_point - start_point  # (2, )
    start_to_end_length_sq = jnp.sum(start_to_end * start_to_end)  # ()

    # Compute vectors from start_point to each obstacle center.
    start_to_obs = obstacles - start_point[None, :]  # (num_obstacles, 2)

    # Compute the projection scalar (t) for each obstacle onto the line SE.
    # This gives the position along SE where the perpendicular from the obstacle falls.
    # A small epsilon is added to avoid division by zero.
    t = jnp.sum(start_to_obs * start_to_end[None, :], axis=1) / (
        start_to_end_length_sq + 1e-8
    )  # (num_obstacles,)

    # Clamp t to the interval [0, 1] so that we only consider the segment.
    t = jnp.clip(t, 0.0, 1.0)

    # Compute the closest point on the segment for each obstacle.
    # Broadcasting t to scale the vector SE for each obstacle.
    closest_points = start_point[None, :] + t[:, None] * start_to_end[None, :]  # (num_obstacles, 2)

    # Compute the squared distances from each obstacle center to its closest point on the segment.
    dist_sq = jnp.sum((closest_points - obstacles) ** 2, axis=1)

    return jnp.any(dist_sq <= effective_rad**2)


@struct.dataclass
class RRTStarState(RRTState):
    key: Array
    pos: Array  # (num_samples, num_agents, [x, y])
    parent: Array  # (num_samples, num_agents)
    cost: Array  # (num_samples, num_agents)
    goal_reached: Array  # (num_agents, )
    idx: int


class RRTStar:
    def __init__(
        self, env, num_samples, step_size, num_neighbours=20, goal_rad=None, collision_func=check_collision
    ):
        self.num_samples = num_samples
        self.step_size = step_size

        assert env.homogeneous_agents, "RRTStar only supports homogeneous agents"
        assert env.homogeneous_landmarks, "RRTStar only supports homogeneous landmarks"
        assert env.homogeneous_goals, "RRTStar only supports homogeneous goals"

        if goal_rad is None:
            if env.homogeneous_goals:
                self.goal_rad = env.map_generator.goal_rad
            else:
                self.goal_rad = env.map_generator.agent_rad_range[0] / 2.5
        else:
            self.goal_rad = goal_rad

        if env.homogeneous_agents:
            self.agent_rad = env.map_generator.agent_rad
        else:
            self.agent_rad = env.map_generator.agent_rad_range[0]

        self.num_agents = env.num_agents

        self.effective_rad = self.agent_rad + env.map_generator.landmark_rad
        self.sample_limit = jnp.array([env.map_generator.width / 2, env.map_generator.height / 2])

        self.check_collision = collision_func

        self.num_neighbours = num_neighbours
        self.agent_indices = jnp.arange(self.num_agents)

    def run(
        self,
        key: Array,
        start: Array,
        goal: Array,
        landmark_pos: Array,
    ):
        """
        Run RRT algorithm to find a path between start and goal
        start (num_agents, 2)
        goal (num_agents, 2)
        landmark_pos (num_landmarks, 2)
        """

        def _condition(state: RRTState):
            # return (state.idx < self.num_samples) & ~jnp.all(state.goal_reached)
            return state.idx < self.num_samples

        def _step(state: RRTStarState):
            """
            :param state: The `state` parameter in the `_step` function represents the current state of the
            RRT.
            :type state: RRTStarState
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
            distances = jnp.where((jnp.arange(self.num_samples) < state.idx)[:, None], distances, jnp.inf)
            distances = jnp.where(state.parent == -2.0, jnp.inf, distances)  # -2 means invalid
            closest_idx = jnp.argmin(distances, axis=0)  # (num_agents, )

            tree_pos = state.pos[
                closest_idx, self.agent_indices, :
            ]  # (num_agents, 2) - nearest tree pos to connect with)

            direction = sampled_pos - tree_pos
            direction_dist = distances[closest_idx, self.agent_indices]  # (num_agents, )

            clipped_direction = jnp.where(
                (direction_dist > self.step_size)[:, None],
                direction / direction_dist[:, None] * self.step_size,
                direction,
            )  # (num_agents, 2)
            test_pos = tree_pos + clipped_direction  # (num_agents, 2)

            # Check free space, line collision
            valid = ~self.check_collision(
                tree_pos, test_pos, landmark_pos, self.effective_rad
            )  # (num_agents, )

            # find neighbours
            tree_distances = jnp.linalg.norm(
                state.pos[:, :, :] - test_pos[None, :, :], axis=-1
            )  # (rrt_samples, num_agents)
            tree_distances = jnp.where(
                (jnp.arange(self.num_samples) < state.idx)[:, None], tree_distances, jnp.inf
            )
            tree_distances = jnp.where(state.parent == -2.0, jnp.inf, tree_distances)

            k_neighbours_idx = jnp.argpartition(tree_distances, self.num_neighbours, axis=0)[
                : self.num_neighbours, :
            ]  # (num_neighbours, num_agents)
            k_neighbours_pos = state.pos[
                k_neighbours_idx, self.agent_indices, :
            ]  # (num_neighbours, num_agents, 2)

            valid_neighbours = ~jax.vmap(self.check_collision, in_axes=[0, None, None, None])(
                k_neighbours_pos, test_pos, landmark_pos, self.effective_rad
            )  # (num_neighbours, num_agents)
            valid_neighbours = jnp.where(
                k_neighbours_idx < state.idx, valid_neighbours, False
            )  # (num_neighbours, num_agents)
            valid_neighbours = valid[None, :] & valid_neighbours  # (num_neighbours, num_agents)

            neighbour_cost = (
                state.cost[k_neighbours_idx, self.agent_indices]
                + tree_distances[k_neighbours_idx, self.agent_indices]
            )  # (num_neighbours, num_agents)
            neighbour_cost = jnp.where(
                valid_neighbours, neighbour_cost, jnp.inf
            )  # (num_neighbours, num_agents)

            best_neighbour_rel_idx = jnp.argmin(neighbour_cost, axis=0)  # (num_agents, )
            best_neighbour_cost = neighbour_cost[best_neighbour_rel_idx, self.agent_indices]  # (num_agents, )
            best_neighbour_idx = k_neighbours_idx[
                best_neighbour_rel_idx, self.agent_indices
            ]  # (num_agents, )
            valid_best_neighbour = (
                valid & valid_neighbours[best_neighbour_rel_idx, self.agent_indices]
            )  # (num_agents, )

            # new_pos = jnp.where(valid_best_neighbour[:, None], state.pos[best_neighbour_idx, self.agent_indices, :], -2.0)
            new_pos = jnp.where(valid_best_neighbour[:, None], test_pos, 0.0)  # (num_agents, 2)
            new_parent = jnp.where(valid_best_neighbour, best_neighbour_idx, -2)  # (num_agents, )
            new_cost = jnp.where(valid_best_neighbour, best_neighbour_cost, -2.0)  # (num_agents, )

            state = state.replace(
                pos=state.pos.at[state.idx].set(new_pos),
                parent=state.parent.at[state.idx].set(new_parent),
                cost=state.cost.at[state.idx].set(new_cost),
            )

            # rewire neighbours
            new_neighbour_cost = (
                best_neighbour_cost[None, :] + tree_distances[k_neighbours_idx, self.agent_indices]
            )  # (num_neighbours, num_agents)
            new_neighbour_cost = jnp.where(
                valid_neighbours, new_neighbour_cost, jnp.inf
            )  # (num_neighbours, num_agents)

            rewire = (
                new_neighbour_cost < state.cost[k_neighbours_idx, self.agent_indices]
            )  # (num_neighbours, num_agents)
            rewire = jnp.where(valid_neighbours, rewire, False)  # (num_neighbours, num_agents)

            new_parent_for_rewire = jnp.full((self.num_agents,), state.idx, dtype=jnp.int32)
            new_neighbour_parent = jnp.where(
                rewire, new_parent_for_rewire[None, :], state.parent[k_neighbours_idx, self.agent_indices]
            )  # (num_neighbours, num_agents)
            new_neighbour_cost = jnp.where(
                rewire, new_neighbour_cost, state.cost[k_neighbours_idx, self.agent_indices]
            )  # (num_neighbours, num_agents)

            state = state.replace(
                parent=state.parent.at[k_neighbours_idx, self.agent_indices].set(new_neighbour_parent),
                cost=state.cost.at[k_neighbours_idx, self.agent_indices].set(new_neighbour_cost),
            )

            goal_just_reached = jnp.linalg.norm(test_pos - goal, axis=-1) < self.goal_rad  # (num_agents, )
            goal_just_reached = jnp.where(valid_best_neighbour, goal_just_reached, False)
            new_goal_reached = jnp.where(state.goal_reached, True, goal_just_reached)

            state = state.replace(
                key=new_key,
                pos=state.pos.at[state.idx].set(new_pos),
                parent=state.parent.at[state.idx].set(new_parent),
                goal_reached=new_goal_reached,
                # idx = state.idx + 1 * jnp.any(valid),
                idx=state.idx + 1,
            )

            return state

        state = RRTStarState(
            key=key,
            pos=jnp.full((self.num_samples, self.num_agents, 2), start, dtype=jnp.float32),
            parent=jnp.full((self.num_samples, self.num_agents), -1.0, dtype=jnp.int32),
            cost=jnp.full((self.num_samples, self.num_agents), 0.0, dtype=jnp.float32),
            goal_reached=jnp.linalg.norm(start - goal, axis=-1) < self.goal_rad,
            idx=1,
        )

        res_state = jax.lax.while_loop(_condition, _step, state)
        return res_state

    def find_last_idx(self, state: RRTState):
        idx = jnp.arange(self.num_samples)
        valid_idx = jnp.where((state.parent != -1) & (state.parent != -2), idx[:, None], -1)
        return jnp.max(valid_idx, axis=0)

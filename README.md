<p align="center">
<img src="images/string_grid_camar.svg" width="300" height="300" />
</p>

# 🦟 CAMAR
CAMAR (Continuous Action Multi-Agent Routing) Benchmark is a fast, GPU-accelerated environment for multi-agent navigation and collision avoidance tasks in continuous state and action spaces. Designed to bridge the gap between multi-robot systems and MARL research, CAMAR emphasizes efficient simulation speeds (exceeding 100K+ Steps Per Second) and evaluation protocols to evaluate agent navigation capabilities.

# Installation

Camar can be installed from PyPi (will be available after publication):

```bash
pip install camar
```

By default the installation includes a CPU-only version of JAX, to install a CUDA version:

```bash
pip install camar[cuda12]
```
or

```bash
pip install jax[cuda12] camar
```

If you want another version of JAX (i.e. TPU), you will need to install it separately, following the [JAX documentaion](https://docs.jax.dev/en/latest/installation.html).

Additionally, there are several options you may want to install:
```bash
# To use CAMAR as a TorchRL environment
pip install camar[torchrl]

# To enable matplotlib visualisation (by default only SVG)
pip install camar[matplotlib]

# To use LabMaze maps
pip install camar[labmaze]

# To use MovingAI maps
pip install camar[movingai]

# To train baselines in BenchMARL
pip install camar[benchmarl]
```

# Quickstart

Camar interface is close to other jax-based RL envirionments and stays close to the gymnax interface:

```python
import jax
from camar import camar_v0


key = jax.random.key(0)
key, key_r, key_a, key_s = jax.random.split(key, 4)

# Create an environment with a random_grid map by default
env = camar_v0()

# Reset the environment
obs, state = env.reset(key_r)

# Sample random actions
actions = env.action_spaces.sample(key_a)

# Step the environment
obs, state, reward, done, info = env.step(key_s, state, actions)
```

You can use it in jax vectorized manner:
```python
import jax
from camar import camar_v0

key = jax.random.key(0)
key, key_r, key_a, key_s = jax.random.split(key, 4)

# Set the number of parallel vectorized environments
num_envs = 1000

# Create an environment
env = camar_v0()

# Set vectorized functions for action sampling, env reset, env step
action_sampler = jax.jit(jax.vmap(env.action_spaces.sample, in_axes=[0, ]))
env_reset_fn = jax.jit(jax.vmap(env.reset, in_axes=[0, ]))
env_step_fn = jax.jit(jax.vmap(env.step, in_axes=[0, 0, 0, ]))

# Generate jax random keys for each parallel environment
key_r = jax.numpy.vstack(jax.random.split(key_r, num_envs))
key_a = jax.numpy.vstack(jax.random.split(key_a, num_envs))
key_s = jax.numpy.vstack(jax.random.split(key_s, num_envs))

# perform reset and steps as usual
...
```

For the ease of use we have also adapted [wrappers from Craftax Baselines](https://github.com/MichaelTMatthews/Craftax_Baselines/blob/main/wrappers.py)
```python
from camar import camar_v0
from camar.wrappers import BatchEnvWrapper, AutoResetEnvWrapper, OptimisticResetVecEnvWrapper


num_envs = 1000
env = OptimisticResetVecEnvWrapper(env=camar_v0(), num_envs=num_envs, reset_ratio=200)
```

# Maps
Default map is `random_grid` with random positions of obstacles, agents and goals on every `env.reset`, but there are variety of maps available in CAMAR. All of them can be imported from `camar.maps` or just use the map name with creating an env. Here is an example:
```python
from camar.maps import string_grid, movingai, labmaze_grid
from camar import camar_v0


map_str = """
.....#.....
.....#.....
...........
.....#.....
.....#.....
#.####.....
.....###.##
.....#.....
.....#.....
...........
.....#.....
"""

string_grid_map = string_grid(map_str=map_str, num_agents=8)
random_grid_map = random_grid()
labmaze_map = labmaze_grid(num_maps=10)

env1 = camar_v0(string_grid_map)
env2 = camar_v0(random_grid_map)
env3 = camar_v0(labmaze_map)

# or simply
env1 = camar_v0("string_grid", map_str=map_str, num_agents=8)
env2 = camar_v0("random_grid", )
env3 = camar_v0("labmaze_grid", num_maps=10)
```

Below you can find the list of all available maps and map_kwargs for each ([random_grid](#random_grid), [string_grid](#string_grid), [batched_string_grid](#batched_string_grid), [labmaze_grid](#labmaze_grid), [movingai](#movingai), [caves_cont](#caves_cont)):
| Map name                                                  | **map_kwargs                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | Visualization of 2 env resets with different seeds                                                                                                                                                                                                                                                                                                                                                                                                                     |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| <div name="random_grid">random_grid</div>                 | `num_rows: int = 20` - number of rows.<br>`num_cols: int = 20` - number of columns.<br>`obstacle_density: float = 0.2` - obstacle density.<br>`num_agents: int = 32` - number of agents.<br>`obstacle_size: float = 0.4` - size of each obstacle.<br>`grain_factor: int = 3` - number of circles per obstacle edge.<br>`obstacle_size: float = 0.4` - the size of each square-like obstacle.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | <div style="position: relative; width: 250px; height: 502px;"> <img src="images/random_grid_1.svg" style="position: absolute; top: 0; left: 0; width: 250px; height: 250px; opacity: 1.0;" /> <div style="position: absolute; top: 250; left: 0; width: 250px; height: 2px; background-color: #ff2d00;"></div> <img src="images/random_grid_2.svg" style="position: absolute; top: 252; left: 0; width: 250px; height: 250px; opacity: 1.0;" /> </div>                 |
| <div name="string_grid">string_grid</div>                 | `map_str: str` - string layout of a grid map. "." is a free cell, otherwise it is an obstacle.<br>`free_pos_str: Optional[str] = None` - to force agents and goals be generated on certain parts of maps. "." is a free cell on which agents and goals can be generated (if is free according to map_str).<br> `agent_idx: Optional[ArrayLike] = None,` - jnp.array([row_id, col_id]) of desired cells on the map_str for initial agent positions.<br>`goal_idx: Optional[ArrayLike] = None` - similar to agent_idx but for goal positions.<br>`num_agents: int = 10` - number of agents.<br>`random_agents: bool = True` - whether generate new agent positions each env.reset or randomly initialize and use them.<br>`random_goals: bool = True` - similar to random_agents but for goal positions.<br>`remove_border: bool = False` - flag whether borders should be deleted or not.<br>`add_border: bool = True` - flag whether additional borders should be added or not.<br>`obstacle_size: float = 0.1` - size of each circle obstacle.<br>`agent_size: float = 0.09` - size of each agent.<br>`max_free_pos: Optional[int] = None` - the maximum amount of free positions for generating agents and goals. this will randomly truncate possible free positions. Can be used for memory control.<br>`map_array_preprocess: callable = lambda map_array: map_array` - preprocess function for map_str after converting it to an array format. Can be used for resizing.<br>`free_pos_array_preprocess: callable = lambda free_pos_array: free_pos_array` - similar to map_array_preproocess, but for free_pos_str. | <div style="position: relative; width: 250px; height: 502px;"> <img src="images/string_grid_1.svg" style="position: absolute; top: 0; left: 0; width: 250px; height: 250px; opacity: 1.0;" /> <div style="position: absolute; top: 250; left: 0; width: 250px; height: 2px; background-color: #ff2d00;"></div> <img src="images/string_grid_2.svg" style="position: absolute; top: 252; left: 0; width: 250px; height: 250px; opacity: 1.0;" /> </div>                 |
| <div name="batched_string_grid">batched_string_grid</div> | Has the same kwargs as `string_grid`, but `map_str_batch`, `free_pos_str_batch`, `agent_idx_batch`, `goal_idx_batch` - list of `map_str`, `free_pos_str`, `agent_idx`, `goal_idx`, respectively.<br>**Note:** If you want to use `map_str_batch` with different map sizes, you must resize them manually or provide `map_array_preprocess` with resizing procedure; the same for `free_pos_str_batch`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | <div style="position: relative; width: 250px; height: 502px;"> <img src="images/batched_string_grid_1.svg" style="position: absolute; top: 0; left: 0; width: 250px; height: 250px; opacity: 1.0;" /> <div style="position: absolute; top: 250; left: 0; width: 250px; height: 2px; background-color: #ff2d00;"></div> <img src="images/batched_string_grid_3.svg" style="position: absolute; top: 252; left: 0; width: 250px; height: 250px; opacity: 1.0;" /> </div> |
| <div name="labmaze_grid">labmaze_grid</div>               | `num_maps: int` - the number of maps to generate and batch.<br>`height: int = 11` - height of the grid map.`width: int = 11` - width of the grid map.<br>`max_rooms: int = -1` - the maximum number of rooms on the map.<br>`seed: int = 0` - seed for generation.<br>`num_agents: int = 10` - the number of agents.<br>`obstacle_size: float = 0.1` - the size of each circle obstacle.<br>`agent_size: float = 0.06` - size of each agent.<br>`max_free_pos: int = None` - the maximum number of free positions.<br>`**labmaze_kwargs` - all other kwargs of labmaze.RandomGrid.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | <div style="position: relative; width: 250px; height: 502px;"> <img src="images/labmaze_grid_0.svg" style="position: absolute; top: 0; left: 0; width: 250px; height: 250px; opacity: 1.0;" /> <div style="position: absolute; top: 250; left: 0; width: 250px; height: 2px; background-color: #ff2d00;"></div> <img src="images/labmaze_grid_3.svg" style="position: absolute; top: 252; left: 0; width: 250px; height: 250px; opacity: 1.0;" /> </div>               |
| <div name="movingai">movingai</div>                       | `map_names: List[str]` - list of map names from MovingAI 2D Benchmark (example: `map_names=["street/Denver_0_1024", "bg_maps/AR0072SR", ...]`). All maps will be downloaded to ".cache/movingai/".<br>`height: int = 128` - all maps are resized to this height.<br>`width: int = 128` - all maps will be resized to this width.<br>`low_thr: float = 3.7` - threshold for edge detection.<br>`only_edges: bool = True` - whether detect edges and use them or not.<br>`remove_border: bool = True` - whether borders should be deleted or not.<br>`add_border: bool = False` - whether additional borders should be added or not.<br>`num_agents: int = 10` - the number of agents.<br>`obstacle_size: float = 0.1` - size of each circle obstacle.<br>`agent_size: float = 0.06` - size of each agent.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | <div style="position: relative; width: 250px; height: 502px;"> <img src="images/movingai_0.svg" style="position: absolute; top: 0; left: 0; width: 250px; height: 250px; opacity: 1.0;" /> <div style="position: absolute; top: 250; left: 0; width: 250px; height: 2px; background-color: #ff2d00;"></div> <img src="images/movingai_3.svg" style="position: absolute; top: 252; left: 0; width: 250px; height: 250px; opacity: 1.0;" /> </div>                       |
| <div name="caves_cont">caves_cont</div>                   | `num_rows: int = 128` - the number of rows.<br>`num_cols: int = 128` - the number of columns.<br>`scale: int = 14` - grid factor for gradients in perlin noise. Can be interpreted as a frequency.<br>`landmark_low_ratio: float = 0.55` - left quantile for edges of perlin noise.<br>`landmark_high_ratio: float = 0.72` - right quntile for edges. Only pos that [left, right] becomes landmarks.<br>`free_ratio: int = 0.20` - the same quantile but for the free positions on which agents and goals are generated. Analogue of `max_free_pos`<br> `add_borders: bool = True` - whether to add map borders or not.<br>`num_agents: int = 16` - the number of agents.<br>`obstacle_size: float = 0.1` - obstacle size (circle diameter).<br>`agent_size: float = 0.2` - agent size (circle diameter).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | <div style="position: relative; width: 250px; height: 502px;"> <img src="images/caves_cont_2.svg" style="position: absolute; top: 0; left: 0; width: 250px; height: 250px; opacity: 1.0;" /> <div style="position: absolute; top: 250; left: 0; width: 250px; height: 2px; background-color: #ff2d00;"></div> <img src="images/caves_cont_9.svg" style="position: absolute; top: 252; left: 0; width: 250px; height: 250px; opacity: 1.0;" /> </div>                   |
|                                                           |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |


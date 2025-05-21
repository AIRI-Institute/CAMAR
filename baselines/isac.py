import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import dataclasses

import matplotlib.pyplot as plt
import pandas as pd
import torch
from tensordict.nn import TensorDictModule, TensorDictSequential
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import RandomSampler, ReplayBuffer
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal
from torchrl.objectives import SACLoss, SoftUpdate, ValueEstimators
from tqdm.auto import tqdm

import wandb
from camar import camar_v0
from camar.maps import random_grid
from camar.render import SVG_Visualizer
from camar.integrations.torchrl import CamarWrapper

os.chdir(os.getcwd() + "/baselines/")
torch.manual_seed(911)

model_name = "ISAC"

device = torch.device("cuda:0")

frames_per_batch = 54_000
n_iters = 150
total_frames = frames_per_batch * n_iters

# Replay buffer
memory_size = 1_000_000

n_optimiser_steps = 100
train_batch_size = 5000
lr = 3e-4
max_grad_norm = 1.0

gamma = 0.99
polyak_tau = 0.005



num_rows = 10
num_cols = 10
obstacle_density = 0.0
num_agents = 5
grain_factor = 6

window = 0.8

map_generator = random_grid(
    num_rows = num_rows,
    num_cols = num_cols,
    obstacle_density = obstacle_density,
    num_agents = num_agents,
    grain_factor = grain_factor,
)

env = camar_v0(
    map_generator = map_generator,
    window = window,
    lifelong = False,
    max_steps = 900,
    contact_force=500,
    contact_margin=1e-3,
    dt=0.01,
    frameskip=7,
    max_obs=8,
)

num_envs = frames_per_batch // env.max_steps
print("num_envs =", num_envs)

env = CamarWrapper(env=env, device=device, batch_size=[num_envs], seed=0)

env = TransformedEnv(
    env,
    RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]),
)

# set up policy

policy_net = torch.nn.Sequential(
    MultiAgentMLP(
        n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
        n_agent_outputs=2 * env.action_spec.shape[-1],
        n_agents=num_agents,
        device=device,
        depth=2,
        num_cells=256,
        activation_class=torch.nn.Tanh,
        share_params=True,
        centralized=False,
        # layer_class=NoisyLinear, #doesn't work
    ),
    NormalParamExtractor(),
)

policy_module = TensorDictModule(
    policy_net,
    in_keys=[("agents", "observation")],
    out_keys=[("agents", "loc"), ("agents", "scale")],
)

policy = ProbabilisticActor(
    module=policy_module,
    spec=env.full_action_spec["agents", "action"],
    in_keys=[("agents", "loc"), ("agents", "scale")],
    out_keys=[env.action_key],
    distribution_class=TanhNormal,
    distribution_kwargs={
        "low": env.action_spec.space.low,
        "high": env.action_spec.space.high,
    },
    return_log_prob=True,
    log_prob_key=("agents", "log_prob"),
)

# set up q network

cat_module = TensorDictModule(
    lambda obs, action: torch.cat([obs, action], dim=-1),
    in_keys=[("agents", "observation"), ("agents", "action")],
    out_keys=[("agents", "obs_action")],
)

qvalue_net = MultiAgentMLP(
    n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1] + env.full_action_spec["agents", "action"].shape[-1],
    n_agent_outputs=1,
    n_agents=num_agents,
    device=device,
    depth=2,
    num_cells=256,
    activation_class=torch.nn.Tanh,
    share_params=True, # can be changed
    centralized=False, # True for masac and False for isac
)

qvalue_module = TensorDictModule(
    module=qvalue_net,
    in_keys=[("agents", "obs_action")],
    out_keys=[("agents", "state_action_value")],
)

qvalue = TensorDictSequential(
    cat_module, qvalue_module,
)

collector = SyncDataCollector(
    env,
    policy,
    device=device,
    storing_device=device,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
)

replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(
        memory_size, device=device
    ),
    sampler=RandomSampler(),
    batch_size=train_batch_size,
)

loss_module = SACLoss(
    actor_network=policy,
    qvalue_network=qvalue,
    # value_network=value,
)

loss_module.set_keys(
    reward=env.reward_key,
    action=env.action_key,
    log_prob=("agents", "log_prob"),
    state_action_value=("agents", "state_action_value"),
    value=("agents", "state_value"),
    done=("agents", "done"),
    terminated=("agents", "terminated"),
)
loss_module.make_value_estimator(ValueEstimators.TD0, gamma=gamma)

target_updater = SoftUpdate(loss_module, tau=polyak_tau)

optim = torch.optim.Adam(loss_module.parameters(), lr)

def process_batch(batch):
    """
    If the `(group, "terminated")` and `(group, "done")` keys are not present, create them by expanding
    `"terminated"` and `"done"`.
    This is needed to present them with the same shape as the reward to the loss.
    """
    keys = list(batch.keys(True, True))
    group_shape = batch.get_item_shape("agents")
    nested_done_key = ("next", "agents", "done")
    nested_terminated_key = ("next", "agents", "terminated")
    if nested_done_key not in keys:
        batch.set(
            nested_done_key,
            batch.get(("next", "done")).unsqueeze(-1).expand((*group_shape, 1)),
        )
    if nested_terminated_key not in keys:
        batch.set(
            nested_terminated_key,
            batch.get(("next", "terminated"))
            .unsqueeze(-1)
            .expand((*group_shape, 1)),
        )
    return batch

pbar = tqdm(total=n_iters, desc="episode_reward_mean = 0")

wandb.init(
    project="myenv_xppo",
    config={
        "num_rows": num_rows,
        "num_cols": num_cols,
        "obstacle_density": obstacle_density,
        "num_agents": num_agents,
        "grain_factor": grain_factor,
        "frames_per_batch": frames_per_batch,
        "n_iters": n_iters,
        "total_frames": total_frames,
        "n_optimiser_steps": n_optimiser_steps,
        "train_batch_size": train_batch_size,
        "lr": lr,
        "max_grad_norm": max_grad_norm,
        "gamma": gamma,
        "polyak_tau": polyak_tau,
    },
    name=f"{model_name}"
)

plot_data = {
    "env_steps": [],
    "ep_rew_mean": [],
}

env_steps = 0
for iteration, batch in enumerate(collector):
    current_frames = batch.numel()
    batch = process_batch(batch)  # Util to expand done keys if needed

    data_view = batch.reshape(-1) # This just affects the leading dimensions in batch_size of the tensordict
    replay_buffer.extend(data_view)

    for _ in range(n_optimiser_steps):
        subdata = replay_buffer.sample()
        loss_vals = loss_module(subdata)

        loss_total = (
                loss_vals["loss_actor"]
                + loss_vals["loss_alpha"]
                + loss_vals["loss_qvalue"]
            )

        grad_norms = {}

        loss_total.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            loss_module.parameters(), max_grad_norm
        )

        wandb.log({
                    "loss_actor": loss_vals["loss_actor"].item(),
                    "loss_alpha": loss_vals["loss_alpha"].item(),
                    "loss_qvalue": loss_vals["loss_qvalue"].item(),
                    "loss_total": loss_total.item(),
                    "grad_norm": grad_norm,
        })

        optim.step()
        optim.zero_grad()

        target_updater.step()

    done = batch.get(("next", "done"))[:, :, 0]
    episode_reward_mean = batch.get(("next", "agents", "episode_reward"))[done].mean().item()

    env_steps += data_view.shape[0]

    plot_data["env_steps"].append(env_steps)
    plot_data["ep_rew_mean"].append(episode_reward_mean)

    wandb.log({
        "ep_rew_mean": episode_reward_mean,
        "iter": iteration,
        "env_steps": env_steps,
    })

    pbar.set_description(f"episode_reward_mean = {episode_reward_mean :.2f}", refresh=False)
    pbar.update()

wandb.finish()


plot_df = pd.DataFrame(plot_data)

fig = plt.figure()
ax = fig.add_subplot()

ax.plot(plot_df["env_steps"], plot_df["ep_rew_mean"])
ax.set_xlabel("env_steps")
ax.set_ylabel("ep_rew_mean")

fig.savefig(f"img/ep_rew_mean_{model_name}.png")
plot_df.to_csv(f"data/ep_rew_mean_{model_name}.csv")


def get_state_from_envs(state, env_id):
    state_data = {
        field.name: getattr(state, field.name)[env_id]
        for field in dataclasses.fields(state)
    }
    return type(state)(**state_data)


def rendering_callback(env, td):
    env.state_seq.append(get_state_from_envs(env._state, 0))


viz_env = camar_v0(
    map_generator = map_generator,
    window = window,
    lifelong = False,
    max_steps = 900,
    contact_force=500,
    contact_margin=1e-3,
    dt=0.01,
    frameskip=7,
    max_obs=8,
)

viz_env = CamarWrapper(env=viz_env, device=device, batch_size=[2], seed=5)

viz_env = TransformedEnv(
    viz_env,
    RewardSum(in_keys=[viz_env.reward_key], out_keys=[("agents", "episode_reward")]),
)


viz_env.state_seq = []

with torch.no_grad():
    out = viz_env.rollout(
        auto_reset=True,
        max_steps=viz_env.max_steps + 1,
        policy=policy,
        callback=rendering_callback,
        auto_cast_to_device=True,
        break_when_any_done=True,
    )


SVG_Visualizer(viz_env._env, viz_env.state_seq).save_svg(f"env_viz/example_{model_name}.svg")

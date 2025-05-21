import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import dataclasses

import matplotlib.pyplot as plt
import pandas as pd
import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal
from torchrl.objectives import ClipPPOLoss, ValueEstimators
from tqdm.auto import tqdm

import wandb
from camar import camar_v0
from camar.maps import random_grid
from camar.render import SVG_Visualizer
from camar.integrations.torchrl import CamarWrapper


os.chdir(os.getcwd() + "/baselines/")
torch.manual_seed(911)

# parameters
params = {
    "CPPO": {
        "policy": {"share_params": True, "centralised": True},
        "critic": {"share_params": True, "centralised": True},
    },
    "MAPPO": {
        "policy": {"share_params": True, "centralised": False},
        "critic": {"share_params": True, "centralised": True},
    },
    "IPPO": {
        "policy": {"share_params": True, "centralised": False},
        "critic": {"share_params": True, "centralised": False},
    },
    "HetIPPO": {
        "policy": {"share_params": False, "centralised": False},
        "critic": {"share_params": False, "centralised": False},
    },
}

model_name = "MAPPO"

policy_params = params[model_name]["policy"]
critic_params = params[model_name]["critic"]

device = torch.device("cuda:0")

frames_per_batch = 200_000
n_iters = 40
total_frames = frames_per_batch * n_iters

num_epochs = 2
minibatch_size = 500
lr = 3e-4
max_grad_norm = 1.0

clip_epsilon = 0.2
gamma = 0.99
lmbda = 0.9
entropy_eps = 1e-4
critic_coef = 1
loss_critic_type = "smooth_l2"

wandb_name = "MAPPO_0.5_on_goal_1.0_collision_1.0_rew_diff"

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
    max_steps = 1000,
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

policy_net = torch.nn.Sequential(
    MultiAgentMLP(
        n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
        n_agent_outputs=2 * env.action_spec.shape[-1],
        n_agents=num_agents,
        device=device,
        depth=2,
        num_cells=256,
        activation_class=torch.nn.Tanh,
        share_params=policy_params["share_params"],
        centralized=policy_params["centralised"],
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
    # spec=env.unbatched_action_spec,
    in_keys=[("agents", "loc"), ("agents", "scale")],
    out_keys=[env.action_key],
    distribution_class=TanhNormal,
    distribution_kwargs={
        "low": env.action_spec.space.low,
        "high": env.action_spec.space.high,
    },
    return_log_prob=True,
    log_prob_key=("agents", "sample_log_prob"),
)

critic_net = MultiAgentMLP(
    n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
    n_agent_outputs=1,
    n_agents=num_agents,
    device=device,
    depth=2,
    num_cells=256,
    activation_class=torch.nn.Tanh,
    share_params=critic_params["share_params"],
    centralized=critic_params["centralised"],
)

critic = TensorDictModule(
    module=critic_net,
    in_keys=[("agents", "observation")],
    out_keys=[("agents", "state_value")],
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
    storage=LazyTensorStorage(frames_per_batch, device=device),
    sampler=SamplerWithoutReplacement(),
    batch_size=minibatch_size,
)

loss_module = ClipPPOLoss(
    actor_network=policy,
    critic_network=critic,
    critic_coef=critic_coef,
    clip_epsilon=clip_epsilon,
    entropy_coef=entropy_eps,
    normalize_advantage=False,
    entropy_bonus=True,
)
loss_module.set_keys(
    reward=env.reward_key,
    action=env.action_key,
    sample_log_prob=("agents", "sample_log_prob"),
    value=("agents", "state_value"),
    done=("agents", "done"),
    terminated=("agents", "terminated"),
)


loss_module.make_value_estimator(ValueEstimators.GAE, gamma=gamma, lmbda=lmbda)
GAE = loss_module.value_estimator

optim = torch.optim.Adam(loss_module.parameters(), lr)

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
        "num_epochs": num_epochs,
        "minibatch_size": minibatch_size,
        "lr": lr,
        "max_grad_norm": max_grad_norm,
        "clip_epsilon": clip_epsilon,
        "gamma": gamma,
        "lmbda": lmbda,
        "entropy_eps": entropy_eps,
        "critic_coef": critic_coef,
        "loss_critic_type": loss_critic_type,
    },
    name=wandb_name,
)

plot_data = {
    "env_steps": [],
    "ep_rew_mean": [],
}

env_steps = 0
for iter, tensordict_data in enumerate(collector):
    tensordict_data.set(
        ("next", "agents", "done"),
        tensordict_data.get(("next", "done"))
        .unsqueeze(-1)
        .expand(tensordict_data.get_item_shape(("next", env.reward_key))),
    )
    tensordict_data.set(
        ("next", "agents", "terminated"),
        tensordict_data.get(("next", "terminated"))
        .unsqueeze(-1)
        .expand(tensordict_data.get_item_shape(("next", env.reward_key))),
    )

    with torch.no_grad():
        GAE(
            tensordict_data,
            params=loss_module.critic_network_params,
            target_params=loss_module.target_critic_network_params,
        )

    data_view = tensordict_data.reshape(-1)
    replay_buffer.extend(data_view)

    for num_epoch in range(num_epochs):
        for num_batch in range(frames_per_batch // minibatch_size):
            num_grad_step = (
                iter * num_epochs * (frames_per_batch // minibatch_size)
                + num_epoch * (frames_per_batch // minibatch_size)
                + num_batch
            )

            subdata = replay_buffer.sample()
            loss_vals = loss_module(subdata)

            loss_value = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + loss_vals["loss_entropy"]
            )

            loss_value.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                loss_module.parameters(), max_grad_norm
            )

            wandb.log(
                {
                    "mean_adv": subdata["advantage"].mean().item(),
                    "loss_obj": loss_vals["loss_objective"].item(),
                    "loss_critic": loss_vals["loss_critic"].item(),
                    "loss_entropy": loss_vals["loss_entropy"].item(),
                    "loss_total": loss_value.item(),
                    "grad_norm": grad_norm,
                    "num_grad_step": num_grad_step,
                }
            )

            optim.step()
            optim.zero_grad()

    collector.update_policy_weights_()

    done = tensordict_data.get(("next", "done"))[:, :, 0]
    episode_reward_mean = (
        tensordict_data.get(("next", "agents", "episode_reward"))[done].mean().item()
    )
    # episode_reward_mean = (
    # 	tensordict_data.get(("next", "agents", "episode_reward")).mean().item()
    # )

    env_steps += data_view.shape[0]

    plot_data["env_steps"].append(env_steps)
    plot_data["ep_rew_mean"].append(episode_reward_mean)

    wandb.log(
        {
            "ep_rew_mean": episode_reward_mean,
            "iter": iter,
            "env_steps": env_steps,
        }
    )

    pbar.set_description(
        f"episode_reward_mean = {episode_reward_mean:.2f}", refresh=False
    )
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

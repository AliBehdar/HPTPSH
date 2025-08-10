import time
import torch
import hydra
import pickle

import logging
import rware
import numpy as np
import gymnasium as gym

from wrappers import *
from pathlib import Path
from model import Policy

from functools import partial
from cpprb import ReplayBuffer
from wrappers import SMACWrapper
from omegaconf import DictConfig

from collections import deque,defaultdict
from torch.utils.tensorboard import SummaryWriter
from ops_utils import compute_clusters,Torcherize
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

wrappers = (
        RecordEpisodeStatistics,
        SquashDones,
        SMACCompatible,
    )

def _compute_returns(storage, next_value,cfg):
    returns = [next_value]
    for rew, done in zip(reversed(storage["rewards"]), reversed(storage["done"])):
        ret = returns[0] * cfg.train.gamma + rew * (1 - done.unsqueeze(1))
        returns.insert(0, ret)

    return returns

def _make_envs(cfg):
    def _env_thunk():
        # print(env_args)
        env = gym.make(cfg.env_name,render_mode="rgb_array", **cfg.env_args)
        if cfg.time_limit:
            env = TimeLimit(env, cfg.time_limit)
        #env = Monitor(env, video_folder="./videos",episode_trigger=lambda _: False )
        for wrapper in wrappers:
            env = wrapper(env)

        return env

    env_thunks = [partial(_env_thunk) for i in range(cfg.train.parallel_envs)]
    if cfg.dummy_vecenv:
        envs = DummyVecEnv(env_thunks)
        envs.buf_rews = np.zeros((cfg.train.parallel_envs, len(envs.observation_space)), dtype=np.float32)
    else:
        envs = SubprocVecEnv(env_thunks, start_method="fork")
    envs = Torcherize(envs,cfg)
    envs = SMACWrapper(envs)
    return envs

def _squash_info(info):
    info = [i for i in info if i]
    new_info = {}
    keys = set([k for i in info for k in i.keys()])
    keys.discard("TimeLimit.truncated")
    for key in keys:
        mean = np.mean([np.array(d[key]).sum() for d in info if key in d])
        new_info[key] = mean
    return new_info

def _log_progress(infos, prev_time, step, cfg, log, writer):
    elapsed = time.time() - prev_time
    ups = cfg.train.log_interval / elapsed
    fps = ups * cfg.train.parallel_envs * cfg.train.n_steps
    mean_reward = np.mean([ep["episode_reward"] for ep in infos])
    battles_won = 100 * np.mean([ep.get("battle_won", 0) for ep in infos])

    # Text logs go to Hydra’s logger
    log.info(f"[Step {step:6d}] Timesteps: {cfg.train.parallel_envs * cfg.train.n_steps * step}")
    log.info(f"UPS: {ups:.1f}, FPS: {fps:.1f}, {100*step/cfg.train.total_steps:.2f}% done")
    log.info(f"Mean reward: {mean_reward:.3f}, Battles won: {battles_won:.1f}%")
    log.info("-" * 50)

    # Scalars go to TensorBoard
    for k, v in _squash_info(infos).items():
        writer.add_scalar(k, v, step)


def _compute_loss(model, storage,cfg):
    with torch.no_grad():
        next_value = model.get_value(storage["state" if cfg.central_v else "obs"][-1])
    returns = _compute_returns(storage, next_value,cfg)

    input_obs = zip(*storage["obs"])
    input_obs = [torch.stack(o)[:-1] for o in input_obs]

    if cfg.central_v:
        input_state = zip(*storage["state"])
        input_state = [torch.stack(s)[:-1] for s in input_state]
    else:
        input_state = None

    input_action_mask = zip(*storage["action_mask"])
    input_action_mask = [torch.stack(a)[:-1] for a in input_action_mask]

    input_actions = zip(*storage["actions"])
    input_actions = [torch.stack(a) for a in input_actions]

    values, action_log_probs, entropy = model.evaluate_actions(
        input_obs, input_actions, input_action_mask, input_state,
    )

    returns = torch.stack(returns)[:-1]
    advantage = returns - values

    actor_loss = (
        -(action_log_probs * advantage.detach()).sum(dim=2).mean()
        - cfg.train.entropy_coef * entropy
    )
    value_loss = (returns - values).pow(2).sum(dim=2).mean()

    loss = actor_loss + cfg.train.value_loss_coef * value_loss
    return loss

@hydra.main(version_base="1.1",config_path="../seps/conf", config_name="configs")
def main(cfg: DictConfig):
    
    torch.set_num_threads(1)

    log = logging.getLogger(__name__)
    log.info("Starting training run")

    tb_dir = Path.cwd() / "tensorboard"
    writer = SummaryWriter(log_dir=tb_dir)
    

    envs = _make_envs(cfg)

    agent_count = len(envs.observation_space)
    obs_size = envs.observation_space[0].shape
    act_size = envs.action_space[0].n

    env_dict = {
        "obs": {"shape": obs_size, "dtype": np.float32},
        "rew": {"shape": 1, "dtype": np.float32},
        "next_obs": {"shape": obs_size, "dtype": np.float32},
        "done": {"shape": 1, "dtype": np.float32},
        "act": {"shape": act_size, "dtype": np.float32},
        "agent": {"shape": agent_count, "dtype": np.float32},
    }
    rb = ReplayBuffer(int(agent_count * cfg.train.pretraining_steps * cfg.train.parallel_envs * cfg.train.n_steps), env_dict)

    # before_add = create_before_add_func(env)

    state_size = envs.get_attr("state_size")[0] if cfg.central_v else None
    clusters = None
    if cfg.train.algorithm_mode.startswith("snac"):
        model_count = 1
    elif cfg.train.algorithm_mode == "iac":
        model_count = len(envs.observation_space)
    elif cfg.train.algorithm_mode == "ops":
        if clusters:
            model_count = clusters
        else:
            model_count = min(10, len(envs.observation_space))

    # make actor-critic model
    model = Policy(envs.observation_space, envs.action_space, cfg.network.architecture, model_count, state_size)
    model.to(cfg.train.device)
    optimizer = torch.optim.Adam(model.parameters(), cfg.train.lr, eps=cfg.train.optim_eps)

    # creates and initialises storage
    obs, state, action_mask = envs.reset()

    storage = defaultdict(lambda: deque(maxlen=cfg.train.n_steps))
    storage["obs"] = deque(maxlen= cfg.train.n_steps + 1)
    storage["done"] = deque(maxlen=cfg.train.n_steps + 1)
    storage["obs"].append(obs)
    storage["done"].append(torch.zeros(cfg.train.parallel_envs))
    storage["info"] = deque(maxlen=10)

    # for smac:
    storage["state"] = deque(maxlen=cfg.train.n_steps + 1)
    storage["action_mask"] = deque(maxlen=cfg.train.n_steps + 1)
    if cfg.central_v:
        storage["state"].append(state)
    storage["action_mask"].append(action_mask)
    # ---------
    storage["actions"]     = deque(maxlen=cfg.train.n_steps)
    storage["rewards"]     = deque(maxlen=cfg.train.n_steps)
    storage["laac_rewards"] = torch.zeros(cfg.train.parallel_envs)
    model.sample_laac(cfg.train.parallel_envs)
    if cfg.train.algorithm_mode == "iac":
        model.laac_sample = torch.arange(len(envs.observation_space)).repeat(cfg.parallel_envs, 1)
        # print(model.laac_sample)
    if cfg.train.algorithm_mode == "ops":
        model.laac_sample = torch.zeros(cfg.train.parallel_envs, agent_count).long()
        # print(model.laac_sample)

    start_time = time.time()
    for step in range(cfg.train.total_steps):
        
        if cfg.train.algorithm_mode == "ops" and step in [cfg.train.delay + cfg.train.pretraining_steps*(i+1) for i in range(cfg.train.pretraining_times)]:
            #print(f"Pretraining at step: {step}")
            cluster_idx = compute_clusters(rb.get_all_transitions(), agent_count,cfg)
            model.laac_sample = cluster_idx.repeat(cfg.train.parallel_envs, 1)
            outdir = Path.cwd()
            with open(Path(outdir) / f"{cfg.env_name}.p", "wb") as f:
                pickle.dump(rb.get_all_transitions(), f)
            


        if step % cfg.train.log_interval == 0 and storage["info"]:
            _log_progress(storage["info"], start_time, step, cfg, log, writer)
            start_time = time.time()
            storage["info"].clear()

        for n_step in range(cfg.train.n_steps):
            with torch.no_grad():
                actions = model.act(storage["obs"][-1], storage["action_mask"][-1])
            (obs, state, action_mask), reward, done, info = envs.step(actions)

            if cfg.train.use_proper_termination:
                bad_done = torch.FloatTensor(
                    [1.0 if i.get("TimeLimit.truncated", False) else 0.0 for i in info]
                ).to(cfg.train.device)
                done = done - bad_done

            storage["obs"].append(obs)
            storage["actions"].append(actions)
            storage["rewards"].append(reward)
            storage["done"].append(done)
            storage["info"].extend([i for i in info if "episode_reward" in i])
            storage["laac_rewards"] += reward

            if cfg.train.algorithm_mode == "ops" and step < cfg.train.delay + cfg.train.pretraining_times * cfg.train.pretraining_steps:
                for agent in range(len(obs)):

                    one_hot_action = torch.nn.functional.one_hot(actions[agent], act_size).squeeze().numpy()
                    one_hot_agent = torch.nn.functional.one_hot(torch.tensor(agent), agent_count).repeat(cfg.train.parallel_envs, 1).numpy()

                    if bad_done[0]:
                        nobs = info[0]["terminal_observation"]
                        nobs = [torch.tensor(o).unsqueeze(0) for o in nobs]
                    else:
                        nobs = obs
                        
                    data = {
                        "obs": storage["obs"][-2][agent].numpy(),
                        "act": one_hot_action,
                        "next_obs": nobs[agent].numpy(),
                        "rew":  reward[:, agent].unsqueeze(-1).numpy(),
                        "done": done[:].unsqueeze(-1).numpy(),
                        # "policy": np.array([model.laac_sample[0, agent].float().item()]),
                        "agent": one_hot_agent,
                        # "timestep": step,
                        # "nstep": n_step,
                    }
                    rb.add(**data)

            # for smac:
            if cfg.central_v:
                storage["state"].append(state)

            storage["action_mask"].append(action_mask)
            # ---------

        if cfg.train.algorithm_mode == "ops" and step < cfg.train.pretraining_steps and cfg.train.delay_training:
            continue

        loss = _compute_loss(model, storage)

        # if laac_mode=="laac" and step and step % laac_timestep == 0:
        #     loss += _compute_laac_loss(model, storage)
        #if step %1000==0:  
            #scalar_loss = loss.item() 
            #print("Step:",step," Total steps ",cfg.train.total_steps,"And loss",scalar_loss)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
    envs.close()
    writer.close()

if __name__=="__main__":
    main()
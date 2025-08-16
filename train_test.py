import logging,hydra
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
from ops_utils import compute_clusters,Torcherize,plot_training
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
        env = gym.make("rware:rware-tiny-4ag-v2",render_mode="rgb_array",disable_env_checker=True, **cfg.env_args)
        if cfg.time_limit:
            env = TimeLimit(env, cfg.time_limit)
        env =PickupRewardWrapper(env)
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
    env_list = [list(item.values())[0] for item in cfg.env]
    print(env_list[0])
    for env in env_list:
        envs = _make_envs(cfg)


        agent_count = len(envs.observation_space)
        obs_size = envs.observation_space[0].shape
        act_size = envs.action_space[0].n+1
        print('envs.action_space[0].n',envs.action_space[0].n+1)
        env_dict = {
            "obs": {"shape": obs_size, "dtype": np.float32},
            "rew": {"shape": 1, "dtype": np.float32},
            "next_obs": {"shape": obs_size, "dtype": np.float32},
            "done": {"shape": 1, "dtype": np.float32},
            "act": {"shape": act_size, "dtype": np.float32},
            "agent": {"shape": agent_count, "dtype": np.float32},
        }
        # Adjusted buffer size for episode-based; assumes cfg.train.average_episode_length is set in config
        rb = ReplayBuffer(int(agent_count * cfg.train.pretraining_episodes * cfg.train.parallel_envs * cfg.train.average_episode_length), env_dict)
        #
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
        model = Policy(envs.observation_space, envs.action_space, cfg.network.architecture, model_count, state_size,cfg)
        model.to(cfg.train.device)
        optimizer = torch.optim.Adam(model.parameters(), cfg.train.lr, eps=cfg.train.optim_eps)

        # creates and initialises storage
        obs, state, action_mask = envs.reset()

        # for smac:
        if cfg.central_v:
            pass  # state is used
        # ---------
        model.sample_laac(cfg.train.parallel_envs)
        if cfg.train.algorithm_mode == "iac":
            model.laac_sample = torch.arange(len(envs.observation_space)).repeat(cfg.train.parallel_envs, 1)
            # print(model.laac_sample)
        if cfg.train.algorithm_mode == "ops":
            model.laac_sample = torch.zeros(cfg.train.parallel_envs, agent_count).long()
            # print(model.laac_sample)

        reward_history = []
        loss_history = []

        plot_dir = Path.cwd() / "plots" 
        total_episodes_completed = 0
        update_step = 0

        laac_rewards = torch.zeros(cfg.train.parallel_envs, agent_count).to(cfg.train.device)  # assuming device

        while total_episodes_completed < cfg.train.total_episodes:
            
            pretrain_points = [cfg.train.delay + cfg.train.pretraining_episodes*(i+1) for i in range(cfg.train.pretraining_times)]
            if cfg.train.algorithm_mode == "ops" and total_episodes_completed in pretrain_points:
                #print(f"Pretraining at episodes: {total_episodes_completed}")
                cluster_idx = compute_clusters(rb.get_all_transitions(), agent_count,cfg)
                model.laac_sample = cluster_idx.repeat(cfg.train.parallel_envs, 1)
                #outdir = Path.cwd()
                #with open(Path(outdir) / f"{cfg.env_name}.p", "wb") as f:
                #    pickle.dump(rb.get_all_transitions(), f)
    
            # Reset storage for this batch as lists
            storage = defaultdict(list)
            storage["obs"].append(obs)
            storage["done"].append(torch.zeros(cfg.train.parallel_envs))
            storage["info"] = []  # list instead of deque
            storage["state"].append(state) if cfg.central_v else None
            storage["action_mask"].append(action_mask)
            storage["actions"] = []
            storage["rewards"] = []

            episodes_this_batch = 0

            while episodes_this_batch < cfg.train.update_every_episodes and total_episodes_completed < cfg.train.total_episodes:
                with torch.no_grad():
                    actions = model.act(storage["obs"][-1], storage["action_mask"][-1])
                    
                (obs, state, action_mask), reward, done, info = envs.step(actions)

                if cfg.train.use_proper_termination:
                    bad_done = torch.FloatTensor(
                        [1.0 if i.get("TimeLimit.truncated", False) else 0.0 for i in info]
                    ).to(cfg.train.device)
                    done = done - bad_done

                storage["actions"].append(actions)
                storage["rewards"].append(reward)
                storage["done"].append(done)
                storage["obs"].append(obs)
                if cfg.central_v:
                    storage["state"].append(state)
                storage["action_mask"].append(action_mask)

                new_episodes = sum(1 for i in info if "episode_reward" in i)
                episodes_this_batch += new_episodes
                total_episodes_completed += new_episodes
                storage["info"].extend([i for i in info if "episode_reward" in i])

                laac_rewards += reward

                if cfg.train.algorithm_mode == "ops" and total_episodes_completed < cfg.train.delay + cfg.train.pretraining_times * cfg.train.pretraining_episodes:
                    for agent in range(agent_count):
                        one_hot_action = torch.nn.functional.one_hot(actions[agent], act_size).numpy()
                        one_hot_agent = torch.nn.functional.one_hot(torch.tensor(agent), agent_count).repeat(cfg.train.parallel_envs, 1).numpy()

                        next_obs = obs[agent].numpy()
                        for env_idx in range(cfg.train.parallel_envs):
                            if bad_done[env_idx]:
                                term_obs = info[env_idx]["terminal_observation"]
                                next_obs[env_idx] = np.array(term_obs[agent])

                        data = {
                            "obs": storage["obs"][-2][agent].numpy(),
                            "act": one_hot_action,
                            "next_obs": next_obs,
                            "rew": reward[:, agent].unsqueeze(-1).numpy(),
                            "done": done.unsqueeze(-1).numpy(),
                            "agent": one_hot_agent,
                        }
                        rb.add(**data)

            if cfg.train.algorithm_mode == "ops" and total_episodes_completed < cfg.train.pretraining_episodes and cfg.train.delay_training:
                storage["info"] = []
                update_step += 1
                continue

            loss = _compute_loss(model, storage,cfg)
            #if reward.sum().item()==1:
            #    print(f"Episodes {total_episodes_completed}, Reward: {reward}")
            rollout_rewards = torch.stack(storage["rewards"])
            mean_reward = torch.mean(rollout_rewards).item()
            reward_history.append(mean_reward)
            loss_history.append(loss.item())

            if total_episodes_completed % 1000 == 0 :  
                scalar_loss = loss.item() 
                print("Episodes completed:",total_episodes_completed," Total episodes ",cfg.train.total_episodes,"And loss",scalar_loss)
                #if reward_history and loss_history and total_episodes_completed % 1000 == 0:
                plot_training(cfg,reward_history, loss_history, total_episodes_completed, plot_dir)
    
            storage["info"] = []
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            update_step += 1
        envs.close()
        writer.close()

if __name__=="__main__":
    main()
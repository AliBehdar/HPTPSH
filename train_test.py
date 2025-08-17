
import torch
import hydra
import logging
import rware
import numpy as np
from pathlib import Path
from model import Policy
from cpprb import ReplayBuffer
from omegaconf import DictConfig
from collections import deque,defaultdict
from torch.utils.tensorboard import SummaryWriter
from ops_utils import compute_clusters,plot_training,compute_loss,make_video,make_envs



@hydra.main(version_base="1.1",config_path="../seps/conf", config_name="configs")
def main(cfg: DictConfig):
    
    torch.set_num_threads(1)

    log = logging.getLogger(__name__)
    log.info("Starting training run")

    tb_dir = Path.cwd() / "tensorboard"
    writer = SummaryWriter(log_dir=tb_dir)
    if cfg.network.hartpart :
        env_list = [list(item.values())[0] for item in cfg.env]
    else:
        env_list =[cfg.env[0]["env1_name"]]
        #print(env_list)
    #print(env_list[0])
    i=0
    for env_name in env_list:
        if len(env_list)==1:
            env_name ="rware:rware-tiny-4ag-v2"
        envs = make_envs(env_name,cfg)

        agent_count = len(envs.observation_space)
        obs_size = envs.observation_space[0].shape
        act_size = envs.action_space[0].n
        #print('envs.action_space[0].n',envs.action_space[0].n+1)
        env_dict = {
            "obs": {"shape": obs_size, "dtype": np.float32},
            "rew": {"shape": 1, "dtype": np.float32},
            "next_obs": {"shape": obs_size, "dtype": np.float32},
            "done": {"shape": 1, "dtype": np.float32},
            "act": {"shape": act_size, "dtype": np.float32},
            "agent": {"shape": agent_count, "dtype": np.float32},
        }
        rb = ReplayBuffer(int(agent_count * cfg.train.pretraining_steps * cfg.train.parallel_envs * cfg.train.n_steps), env_dict)
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
        if i>=1:
            print("model.load_transformer(cfg)")
            model.load_transformer(cfg)
        
        i+=1
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
        #storage["laac_rewards"] = torch.zeros(cfg.train.parallel_envs)
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
        for step in range(cfg.train.total_steps):
            
            if cfg.train.algorithm_mode == "ops" and step in [cfg.train.delay + cfg.train.pretraining_steps*(i+1) for i in range(cfg.train.pretraining_times)]:
                print(f"Pretraining at step: {step}")
                cluster_idx = compute_clusters(rb.get_all_transitions(), agent_count,cfg)
                model.laac_sample = cluster_idx.repeat(cfg.train.parallel_envs, 1)
 
    
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

                if cfg.central_v:
                    storage["state"].append(state)

                storage["action_mask"].append(action_mask)
                # ---------

            if cfg.train.algorithm_mode == "ops" and step < cfg.train.pretraining_steps and cfg.train.delay_training:
                continue

            loss = compute_loss(model, storage,cfg)
            #if reward.sum().item()==1:
            #    print(f"Step {step}, Reward: {reward}")
            rollout_rewards = torch.stack(list(storage["rewards"]))
            mean_reward = torch.mean(rollout_rewards).item()
            reward_history.append(mean_reward)
            loss_history.append(loss.item())

            if step % 1000 == 0 :  
                scalar_loss = loss.item() 
                print("Step:",step," Total steps ",cfg.train.total_steps,"And loss",scalar_loss)
                plot_training(cfg,reward_history, loss_history, step, plot_dir)
    
            storage["info"].clear()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
        r=0
        if r==2:# after making reward show okay
            make_video(cfg,env_name,model,agent_count)

        envs.close()
        writer.close()
        model.save_transformer()

if __name__=="__main__":
    main()
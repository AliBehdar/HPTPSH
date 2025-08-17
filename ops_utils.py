import pickle
import tempfile
import gymnasium as gym
import numpy as np
import torch , os 
from cpprb import ReplayBuffer, create_before_add_func, create_env_dict
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from stable_baselines3.common.vec_env import VecEnvWrapper
from model import LinearVAE
import cv2
from wrappers import *
from functools import partial
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from pathlib import Path
class rbDataSet(Dataset):
    
    def __init__(self, rb,cfg):
        self.rb = rb
        self.data = []
        self.data.append(torch.cat([torch.from_numpy(self.rb[n]) for n in cfg.network.encoder_in], dim=1))
        self.data.append(torch.cat([torch.from_numpy(self.rb[n]) for n in cfg.network.decoder_in], dim=1))
        self.data.append(torch.cat([torch.from_numpy(self.rb[n]) for n in cfg.network.reconstruct], dim=1))
        
        ##print([x.shape for x in self.data])
    def __len__(self):
        return self.data[0].shape[0]
    def __getitem__(self, idx):
        return [x[idx, :] for x in self.data]


def compute_clusters(rb, agent_count,cfg,clusters=None,save_plot=False):
    

    dataset = rbDataSet(rb,cfg)
    
    input_size = dataset.data[0].shape[-1]
    extra_decoder_input = dataset.data[1].shape[-1]
    reconstruct_size = dataset.data[2].shape[-1]
    
    model = LinearVAE(cfg.train.z_features, input_size, extra_decoder_input, reconstruct_size)
    ##print(model)
    model.to(cfg.train.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    # criterion = nn.BCELoss(reduction='sum')
    criterion = nn.MSELoss(reduction="sum")
    def final_loss(bce_loss, mu, logvar):
        """
        This function will add the reconstruction loss (BCELoss) and the 
        KL-Divergence.
        KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        :param bce_loss: recontruction loss
        :param mu: the mean from the latent vector
        :param logvar: log variance from the latent vector
        """
        BCE = bce_loss 
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + cfg.train.kl_weight*KLD

    def fit(model, dataloader):
        model.train()
        running_loss = 0.0
        for encoder_in, decoder_in, y in dataloader:
            optimizer.zero_grad()
            reconstruction, mu, logvar = model(encoder_in, decoder_in)
            bce_loss = criterion(reconstruction, y)
            loss = final_loss(bce_loss, mu, logvar)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_loss = running_loss/len(dataloader.dataset)
        return train_loss

    dataloader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True)

    train_loss = []
    for epoch in tqdm(range(cfg.train.epochs)):
        train_epoch_loss = fit(model, dataloader)
        train_loss.append(train_epoch_loss)

    #print(f"Train Loss: {train_epoch_loss:.6f}")
    x = torch.eye(agent_count)

    with torch.no_grad():
        z = model.encode(x).cpu()
    z = z[:, :]

    if clusters is None:
        clusters = find_optimal_cluster_number(z)
    #print(f"Creating {clusters} clusters.")
    # run k-means from scikit-learn
    kmeans = KMeans(n_clusters=clusters, init='k-means++',n_init=10)
    cluster_ids_x = kmeans.fit_predict(z) # predict labels
    if cfg.train.z_features == 2:
        plot_clusters(kmeans.cluster_centers_, z,cfg)
    return torch.from_numpy(cluster_ids_x).long()


def plot_clusters(cluster_centers, z,cfg):
    plt.figure(figsize=(8, 6))
    if cfg.network.human_selected_idx is None:
        plt.plot(z[:, 0], z[:, 1], 'o')
        plt.plot(cluster_centers[:, 0], cluster_centers[:, 1], 'x')

        for i in range(z.shape[0]):
            plt.annotate(str(i), xy=(z[i, 0], z[i, 1]))

    else:
        colors = 'bgrcmykw'
        for i in range(len(cfg.network.human_selected_idx)):
            plt.plot(z[i, 0], z[i, 1], 'o' + colors[cfg.network.human_selected_idx[i]])

        plt.plot(cluster_centers[:, 0], cluster_centers[:, 1], 'x')
    
    plt.title("Cluster Visualization")
    plt.xlabel("Z[0]")
    plt.ylabel("Z[1]")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("cluster_plot.png")
    print("Cluster plot saved to cluster_plot.png")
    plt.close()
    plt.savefig("cluster.png")

def plot_training(cfg, episode_rewards, loss_history, step, plot_dir):
    import matplotlib.pyplot as plt
    import os

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))  # Two subplots: one for rewards, one for loss

    # Plot Episode Rewards
    episodes = np.arange(1, len(episode_rewards) + 1) if episode_rewards else np.array([])
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward', color='tab:blue')
    if len(episodes) > 0:
        ax1.plot(episodes, episode_rewards, color='tab:blue', label='Episode Reward', linewidth=2)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)  # Add zero line for reference
    ax1.grid(True, alpha=0.3)  # Add grid for better readability
    ax1.legend(loc='upper left')
    ax1.set_title(f'Episode Rewards for Chosen Env (Step {step})')

    # Plot Loss
    steps_loss = np.arange(len(loss_history)) * cfg.train.log_interval
    ax2.set_xlabel('Update Steps')
    ax2.set_ylabel('Loss', color='tab:red')
    ax2.plot(steps_loss, loss_history, color='tab:red', label='Loss', linewidth=2)
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.grid(True, alpha=0.3)  # Add grid for better readability
    ax2.legend(loc='upper right')
    ax2.set_title('Training Loss')

    # Adjust layout
    fig.tight_layout()
    
    # Ensure plot directory exists
    os.makedirs(plot_dir, exist_ok=True)
    
    # Save plot with step-specific filename
    plot_path = os.path.join(plot_dir, f"training_plot_step_{step}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')  # Higher DPI and tight layout
    plt.close()
    
def find_optimal_cluster_number(X):

    range_n_clusters = list(range(2, X.shape[0]))
    scores = {}

    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        scores[n_clusters] = davies_bouldin_score(X, cluster_labels)

    max_key = min(scores, key=lambda k: scores[k])
    return max_key

class Torcherize(VecEnvWrapper):
    
    def __init__(self,venv, cfg):
        super().__init__(venv)
        #self.venv.reset()
        self.cfg=cfg
        self.observe_agent_id = self.cfg.train.algorithm_mode == "snac-a"
        if self.observe_agent_id:
            agent_count = len(self.observation_space)
            self.observation_space = gym.spaces.Tuple(tuple([gym.spaces.Box(low=-np.inf, high=np.inf, shape=((x.shape[0] + agent_count),), dtype=x.dtype) for x in self.observation_space]))

    def reset(self):
        obs = self.venv.reset()
        obs = [torch.from_numpy(o).to(self.cfg.train.device) for o in obs]
        if self.observe_agent_id:
            ids = torch.eye(len(obs)).repeat_interleave(self.cfg.parallel_envs, 0).view(len(obs), -1, len(obs))
            obs = [torch.cat((ids[i], obs[i]), dim=1) for i in range(len(obs))]
        return obs

    def step_async(self, actions):
        actions = [a.squeeze().cpu().numpy() for a in actions]
        actions = list(zip(*actions))
        return self.venv.step_async(actions)

    def step_wait(self):
        obs, rew, done, info = self.venv.step_wait()
        obs = [torch.from_numpy(o).float().to(self.cfg.train.device) for o in obs]
        if self.observe_agent_id:
            ids = torch.eye(len(obs)).repeat_interleave(self.cfg.parallel_envs, 0).view(len(obs), -1, len(obs))
            obs = [torch.cat((ids[i], obs[i]), dim=1) for i in range(len(obs))]

        return (
            obs,
            torch.from_numpy(rew).float().to(self.cfg.train.device),
            torch.from_numpy(done).float().to(self.cfg.train.device),
            info,
        )


def compute_returns(storage, next_value,cfg):
    returns = [next_value]
    for rew, done in zip(reversed(storage["rewards"]), reversed(storage["done"])):
        ret = returns[0] * cfg.train.gamma + rew * (1 - done.unsqueeze(1))
        returns.insert(0, ret)

    return returns


def compute_loss(model, storage,cfg):
    with torch.no_grad():
        next_value = model.get_value(storage["state" if cfg.central_v else "obs"][-1])
    returns = compute_returns(storage, next_value,cfg)

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

def make_video(cfg,env_name,model,agent_count):
            # Save video after training, before saving the model
        video_path = Path.cwd() / f"video/trained_video_{env_name.split(':')[-1]}.mp4"
        frames = []

        # Temporarily set config for single parallel env using DummyVecEnv
        original_parallel_envs = cfg.train.parallel_envs
        original_dummy_vecenv = cfg.dummy_vecenv
        cfg.train.parallel_envs = 1
        cfg.dummy_vecenv = True

        render_envs = make_envs(env_name, cfg)

        # Prepare model for single env rollout
        model.sample_laac(1)
        if cfg.train.algorithm_mode == "iac":
            model.laac_sample = torch.arange(agent_count).repeat(1, 1)
        if cfg.train.algorithm_mode == "ops":
            model.laac_sample = torch.zeros(1, agent_count).long()

        obs, state, action_mask = render_envs.reset()
        frames.append(render_envs.render(mode="rgb_array"))

        done = torch.zeros(1, device=cfg.train.device)
        while not done.any():
            with torch.no_grad():
                actions = model.act(obs, action_mask)
            (obs, state, action_mask), reward, done, info = render_envs.step(actions)
            frames.append(render_envs.render(mode="rgb_array"))

        render_envs.close()

        # Restore original config values
        cfg.train.parallel_envs = original_parallel_envs
        cfg.dummy_vecenv = original_dummy_vecenv

        height, width, _ = frames[0].shape  # Assuming frames are consistent RGB arrays
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec; alternatives: 'XVID' for AVI, 'MJPG' for MJPG
        video_writer = cv2.VideoWriter(str(video_path), fourcc, 10, (width, height))
        for frame in frames:
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR for OpenCV
        video_writer.release()


wrappers = (
        RecordEpisodeStatistics,
        SquashDones,
        SMACCompatible,
    )

def make_envs(env_name,cfg):
    def _env_thunk():
        env = gym.make(env_name,render_mode="rgb_array",disable_env_checker=True, **cfg.env_args)
        if cfg.time_limit:
            env = TimeLimit(env, cfg.time_limit)
        env =PickupRewardWrapper(env)
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
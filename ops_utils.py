import pickle
import tempfile
import gymnasium as gym
import numpy as np
import torch
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


def compute_clusters(rb, agent_count,cfg,save_plot=False):
    

    dataset = rbDataSet(rb,cfg)
    
    input_size = dataset.data[0].shape[-1]
    extra_decoder_input = dataset.data[1].shape[-1]
    reconstruct_size = dataset.data[2].shape[-1]
    
    model = LinearVAE(cfg.z_features, input_size, extra_decoder_input, reconstruct_size)
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
        for encoder_in, decoder_in, y in enumerate(dataloader):
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

    print(f"Train Loss: {train_epoch_loss:.6f}")
    x = torch.eye(agent_count)

    with torch.no_grad():
        z = model.encode(x).cpu()
    z = z[:, :]

    if clusters is None:
        clusters = find_optimal_cluster_number(z)
    print(f"Creating {clusters} clusters.")
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
    # plt.savefig("cluster.png")

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
        obs = [torch.from_numpy(o).to(self.cfg.device) for o in obs]
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
        obs = [torch.from_numpy(o).float().to(self.cfg.device) for o in obs]
        if self.observe_agent_id:
            ids = torch.eye(len(obs)).repeat_interleave(self.cfg.parallel_envs, 0).view(len(obs), -1, len(obs))
            obs = [torch.cat((ids[i], obs[i]), dim=1) for i in range(len(obs))]

        return (
            obs,
            torch.from_numpy(rew).float().to(self.cfg.device),
            torch.from_numpy(done).float().to(self.cfg.device),
            info,
        )



from collections import deque
from time import perf_counter
import torch
import numpy as np
from gymnasium import ObservationWrapper, spaces
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from stable_baselines3.common.vec_env import VecEnvWrapper

class RecordEpisodeStatistics(gym.Wrapper):
    """ Multi-agent version of RecordEpisodeStatistics gym wrapper"""

    def __init__(self, env, deque_size=100):
        super().__init__(env)
        try:
            n_agents = len(env.observation_space)
        except TypeError:
            n_agents = getattr(env, "n_agents", 1)
        self.n_agents = n_agents

        self.t0 = perf_counter()
        self.episode_reward = np.zeros(self.n_agents, dtype=np.float64)
        self.episode_length = 0
        self.reward_queue = deque(maxlen=deque_size)
        self.length_queue = deque(maxlen=deque_size)

    def reset(self, **kwargs):
        observation = super().reset(**kwargs)
        #self.episode_reward = 0
        self.episode_reward[:] = 0.0
        self.episode_length = 0
        self.t0 = perf_counter()

        return observation
    
    def step(self, action):
        out = super().step(action)
        # unpack 5-tuple vs. 4-tuple
        if len(out) == 5:
            obs, rew, terminated, truncated, info = out
        else:
            obs, rew, done4, info = out
            terminated = ([done4] * self.n_agents
                          if isinstance(done4, bool)
                          else list(done4))
            truncated  = [False] * self.n_agents

        # ensure lists
        if not isinstance(terminated, (list, tuple)):
            terminated = [terminated]
        if not isinstance(truncated, (list, tuple)):
            truncated = [truncated]

        done_list = [t or tr for t, tr in zip(terminated, truncated)]

        # accumulate rewards and length
        rew_list = rew.tolist() if isinstance(rew, np.ndarray) else list(rew)
        for i, r in enumerate(rew_list):
            self.episode_reward[i] += float(r)
        self.episode_length += 1

        # on episode end, log and reset counters
        if all(done_list):
            info["episode_reward"] = list(self.episode_reward)
            info["episode_length"] = self.episode_length
            info["episode_time"]   = perf_counter() - self.t0
            self.reward_queue.append(list(self.episode_reward))
            self.length_queue.append(self.episode_length)
            # reset counters in-place
            self.episode_reward[:] = 0.0
            self.episode_length = 0
            self.t0 = perf_counter()

        # return 4-tuple
        rew_array = np.array(rew_list, dtype=np.float64)
        return obs, rew_array, terminated, truncated, info

class FlattenObservation(ObservationWrapper):
    r"""Observation wrapper that flattens the observation of individual agents."""

    def __init__(self, env):
        super(FlattenObservation, self).__init__(env)

        ma_spaces = []

        for sa_obs in env.observation_space:
            flatdim = spaces.flatdim(sa_obs)
            ma_spaces += [
                spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(flatdim,),
                    dtype=np.float32,
                )
            ]

        self.observation_space = spaces.Tuple(tuple(ma_spaces))

    def observation(self, observation):
        return tuple(
            [
                spaces.flatten(obs_space, obs)
                for obs_space, obs in zip(self.env.observation_space, observation)
            ]
        )


class SquashDones(gym.Wrapper):
    """Wrapper that squashes multiple dones to a single one using all(dones)"""
    def step(self, action):
        out = self.env.step(action)
        # unpack new vs old API:
        if len(out) == 5:
            obs, rew, terminated, truncated, info = out
        else:
            # legacy 4‑tuple
            obs, rew, done, info = out
            terminated = ([done] 
                          if isinstance(done, bool) 
                          else list(done))
            truncated = [False] * len(terminated)

        # now squash into one done flag per agent
        done_list = [t or tr for t, tr in zip(terminated, truncated)]
        # if you need a single bool (all agents finished):
        done = all(done_list)

        return obs, rew, terminated,truncated, info



class GlobalizeReward(gym.RewardWrapper):
    def reward(self, reward):
        return self.n_agents * [sum(reward)]

class StandardizeReward(gym.RewardWrapper):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stdr_wrp_sumw = np.zeros(self.n_agents, dtype=np.float32)
        self.stdr_wrp_wmean = np.zeros(self.n_agents, dtype=np.float32)
        self.stdr_wrp_t = np.zeros(self.n_agents, dtype=np.float32)
        self.stdr_wrp_n = 0
        
    def reward(self, reward):
        # based on http://www.nowozin.net/sebastian/blog/streaming-mean-and-variance-computation.html
        # update running avg and std
        weight = 1.0

        q = reward - self.stdr_wrp_wmean
        temp_sumw = self.stdr_wrp_sumw + weight
        r = q * weight / temp_sumw
        
        self.stdr_wrp_wmean += r
        self.stdr_wrp_t += q*r*self.stdr_wrp_sumw
        self.stdr_wrp_sumw = temp_sumw
        self.stdr_wrp_n += 1

        if self.stdr_wrp_n == 1:
            return reward

        # calculate standardized reward
        var = (self.stdr_wrp_t * self.stdr_wrp_n) / (self.stdr_wrp_sumw*(self.stdr_wrp_n-1))
        stdr_rew = (reward - self.stdr_wrp_wmean) / (np.sqrt(var) + 1e-6)
        return stdr_rew

class TimeLimit(gym.wrappers.TimeLimit):
    def __init__(self, env, max_episode_steps=None):
        
       
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        # if self.env.spec is not None:
        #     self.env.spec.max_episode_steps = max_episode_steps
        super().__init__(env,max_episode_steps)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        assert (
            self._elapsed_steps is not None
        ), "Cannot call env.step() before calling reset()"
        observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info["TimeLimit.truncated"] = not all(done)
            done = len(observation) * [True]
        return observation, reward, done, info

class PickupRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.pickup_reward = 0.5
        self.prev_carrying = None

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.prev_carrying = [o[2] for o in obs]  # carrying flag is at index 2 in each agent's observation
        return obs, info

    def step(self, action):
 
        obs, rewards, done, info = self.env.step(action)
        new_carrying = [o[2] for o in obs]  # carrying flag is at index 2
        for i in range(len(new_carrying)):
            if new_carrying[i] == 1 and self.prev_carrying[i] == 0:
                rewards[i] += self.pickup_reward
        self.prev_carrying = new_carrying
        return obs, rewards, done, info

class ClearInfo(gym.Wrapper):
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return observation, reward, done, {}


class SMACCompatible(gym.Wrapper):
    def get_avail_actions(self):
        return [np.ones(x.n) for x in self.action_space]

    def get_state(self):
        return [np.zeros(5) for x in self.observation_space]

class SMACWrapper(VecEnvWrapper):
    def _make_action_mask(self, n_agents):
        action_mask = self.venv.env_method("get_avail_actions")
        #print("this is action_mask"action_mask)
        action_mask = [
            torch.tensor(np.array([avail[i] for avail in action_mask])) for i in range(n_agents)#dghsgh
        ]
        return action_mask

    def _make_state(self, n_agents):
        state = self.venv.env_method("get_state")
        state = torch.from_numpy(np.stack(state))
        return n_agents * [state]

    def reset(self):
        obs = self.venv.reset()
        state = self._make_state(len(obs))
        action_mask = self._make_action_mask(len(obs))
        return obs, state, action_mask

    def step_wait(self):
        obs, rew, done, info = self.venv.step_wait()
        state = self._make_state(len(obs))
        action_mask = self._make_action_mask(len(obs))
        return ((obs, state, action_mask),rew,done,info,)

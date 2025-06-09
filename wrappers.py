"""
A collection of environment wrappers for multi-agent environments
"""
import numpy as np
from collections import deque
from time import perf_counter
import rware
from gymnasium import ObservationWrapper, spaces
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo


class RecordEpisodeStatistics(gym.Wrapper):
    """Multi-agent version of RecordEpisodeStatistics."""

    def __init__(self, env, deque_size=100):
        super().__init__(env)
        # Infer number of agents from the wrapped env:
        try:
            n_agents = len(env.observation_space)
        except TypeError:
            n_agents = getattr(env, "n_agents", 1)
        self.n_agents = n_agents

        # Queues to hold recent episode stats
        self.reward_queue = deque(maxlen=deque_size)
        self.length_queue = deque(maxlen=deque_size)

        # Initialize counters
        self.episode_reward = np.zeros(self.n_agents, dtype=np.float64)
        self.episode_length = 0
        self.t0 = perf_counter()

    def reset(self, **kwargs):
        """Reset env and episode stats."""
        obs, info = super().reset(**kwargs)
        # Reset counters
        self.episode_reward[:] = 0.0
        self.episode_length = 0
        self.t0 = perf_counter()
        return obs, info

    def step(self, action):
        """Step through env, track multi-agent episode stats, and return 5-tuple."""
        out = super().step(action)

        # 1) Normalize to 5-tuple: (obs, rew, terminated, truncated, info)
        if len(out) == 5:
            obs, rew, terminated, truncated, info = out
        else:
            # Old 4-tuple: (obs, rew, done, info)
            obs, rew, done4, info = out
            if isinstance(done4, bool):
                terminated = [done4] * self.n_agents
            else:
                terminated = list(done4)
            truncated = [False] * len(terminated)

        # 2) Combine per-agent flags into a single Python list
        done_list = [t or tr for t, tr in zip(terminated, truncated)]

        # 3) Accumulate rewards & length
        #    Ensure rew is a list of floats
        rew_list = rew.tolist() if isinstance(rew, np.ndarray) else list(rew)
        for i, r in enumerate(rew_list):
            self.episode_reward[i] += float(r)
        self.episode_length += 1

        # 4) If the whole multi-agent episode is done, log stats into `info`
        if all(done_list):
            info["episode_reward"] = list(self.episode_reward)
            for i, r in enumerate(self.episode_reward):
                info[f"agent{i}/episode_reward"] = float(r)
            info["episode_length"] = self.episode_length
            info["episode_time"] = perf_counter() - self.t0

            # Save into history
            self.reward_queue.append(list(self.episode_reward))
            self.length_queue.append(self.episode_length)

        # 5) Return the full 5-tuple for Gymnasium compatibility
        return obs, rew, terminated, truncated, info



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
        # 1) step through the next wrapper/env
        out = super().step(action)

        # 2) normalize Gymnasium 5-tuple vs old 4-tuple
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
        else:
            # old 4-tuple: (obs, reward, done, info)
            obs, reward, done4, info = out
            # build terminated list
            if isinstance(done4, bool):
                n_agents = len(obs)
                terminated = [done4] * n_agents
            else:
                terminated = list(done4)
            # no truncated info in old API
            truncated = [False] * len(terminated)

        # 3) squash per-agent flags to a single boolean per agent for downstream
        #    (but keep both lists around for Gymnasium semantics)
        done_list = [t or tr for t, tr in zip(terminated, truncated)]

        # 4) we return the full 5-tuple so DummyVecEnv can unpack it
        return obs, reward, terminated, truncated, info



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
        
        if max_episode_steps is None and getattr(env,"spec",None) is not None:
            max_episode_steps = env.spec.max_episode_steps
        # if self.env.spec is not None:
        #     self.env.spec.max_episode_steps = max_episode_steps
        super().__init__(env,max_episode_steps)

        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

   # def step(self, action):
    #    assert (
     #       self._elapsed_steps is not None
      #  ), "Cannot call env.step() before calling reset()"
       # observation, reward, done, info= self.env.step(action)
        #self._elapsed_steps += 1
        #if self._elapsed_steps >= self._max_episode_steps:
         #   info["TimeLimit.truncated"] = not all(done)
          #  done = len(observation) * [True]
       # return observation, reward, done, info
    def step(self,action):
        out=super().step(action)

        if len(out) == 5:
            obs, rew, terminated, truncated, info = out
        else:
            # old‐style 4‐tuple
            obs, rew, done4, info = out
            if isinstance(done4, bool):
                n_agents = len(obs)
                terminated = [done4] * n_agents
            else:
                terminated = list(done4)
            truncated = [False] * len(terminated)
        return obs, rew, terminated, truncated, info

class ClearInfo(gym.Wrapper):
    #def step(self, action):
    #    observation, reward, done, info = self.env.step(action)
    #    return observation, reward, done, {}
    def step(self, action):
        out = super().step(action)
        # normalize to 5‐tuple
        if len(out) == 5:
            obs, rew, terminated, truncated, info = out
        else:
            obs, rew, done4, info = out
            if isinstance(done4, bool):
                n_agents = len(obs)
                terminated = [done4] * n_agents
            else:
                terminated = list(done4)
            truncated = [False] * len(terminated)

        # always clear info
        return obs, rew, terminated, truncated, {}


class SMACCompatible(gym.Wrapper):
    def get_avail_actions(self):
        return [np.ones(x.n) for x in self.action_space]

    def get_state(self):
        return [np.zeros(5) for x in self.observation_space]


def Monitor(env, video_folder=None, episode_trigger=None, step_trigger=None,
            reset_keywords=(), info_keywords=(), override_existing=True):
    """
    Drop-in replacement for gym.wrappers.Monitor using Gymnasium’s new wrappers:
      1) Record episode stats (reward/length)
      2) Optionally record video

    Args:
      env:          the base environment
      video_folder: if not None, path to save .mp4 files
      episode_trigger, step_trigger: passed to RecordVideo
      reset_keywords, info_keywords, override_existing: ignored (legacy)
    """
    # 1) always track episode statistics
    env = RecordEpisodeStatistics(env)

    # 2) if video_folder is set, wrap with RecordVideo
    if video_folder is not None:
        env = RecordVideo(
            env,
            video_folder=video_folder,
            episode_trigger=episode_trigger,
            step_trigger=step_trigger
        )
    return env

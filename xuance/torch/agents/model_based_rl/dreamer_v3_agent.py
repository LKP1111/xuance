from typing import Tuple

import torch
from copy import deepcopy

from torch.cuda.amp import autocast

from xuance.common import List, Union, SequentialReplayBuffer
from xuance.environment import DummyVecEnv, SubprocVecEnv
from xuance.torch.agents import OffPolicyAgent
from xuance.torch import REGISTRY_Representation, REGISTRY_Policy
from xuance.torch.utils import ActivationFunctions, dotdict

# '.': import from __init__
from xuance.torch.representations.world_model import DreamerV3WorldModel
from xuance.torch.policies import DreamerV3Policy

import numpy as np
from tqdm import tqdm
import gymnasium as gym
from argparse import Namespace
from xuance.common import Optional


class DreamerV3Agent(OffPolicyAgent):
    def __init__(self,
                 config: Namespace,
                 envs: Union[DummyVecEnv, SubprocVecEnv]):
        super(DreamerV3Agent, self).__init__(config, envs)
        self.config = config = dotdict(vars(config))

        # special judge for atari env
        self.atari = True if self.config.env_name == "Atari" else False

        # continuous or not
        self.is_continuous = (isinstance(self.envs.action_space, gym.spaces.Box))
        self.is_multidiscrete = isinstance(self.envs.action_space, gym.spaces.MultiDiscrete)
        self.config.is_continuous = self.is_continuous  # add to config

        # obs_shape & act_shape
        self.obs_shape = self.observation_space.shape
        """
        hwc 2 chw: 
        agent & memory both uses 'hwc'
        obs needed to be transformed to 'chw' and be normalized before sample & taking an action
        """
        if self.config.pixel:
            self.obs_shape = (self.obs_shape[2], ) + self.obs_shape[:2]
        self.act_shape = self.action_space.n if not self.is_continuous else self.action_space.shape
        self.config.act_shape = self.act_shape  # add to config

        # ratio
        self.replay_ratio = self.config.replay_ratio
        self.current_step, self.gradient_step = 0, 0

        # REGISTRY & create: representation, policy, learner
        ActivationFunctions['silu'] = torch.nn.SiLU
        REGISTRY_Representation['DreamerV3WorldModel'] = DreamerV3WorldModel
        self.models = self._build_representation(representation_key="DreamerV3WorldModel",
                                                config=None, input_space=None)

        REGISTRY_Policy["DreamerV3Policy"] = DreamerV3Policy
        self.policy = self._build_policy()
        self.memory = self._build_memory()
        self.learner = self._build_learner(self.config, self.policy, self.act_shape)

        # train_player & train_states; make sure train & test to be independent
        # self.train_player = self.models.player
        # self.train_player.init_states()
        self.deter_size = config.world_model.rssm.deter_size
        self.stoch_size, self.classes = config.world_model.rssm.stoch_size, config.world_model.rssm.classes
        self.deter = torch.zeros(self.envs.num_envs, self.deter_size).to(config.device)
        self.stoch = torch.zeros(self.envs.num_envs, self.stoch_size, self.classes).to(config.device)
        extra_shape = () if not self.is_continuous else self.act_shape
        self.train_states: List[np.ndarray] = [
            self.envs.buf_obs,  # obs: (envs, *obs_shape),
            np.zeros((self.envs.num_envs,) + extra_shape),  # real_acts
            np.zeros(self.envs.num_envs),  # rews
            np.zeros(self.envs.num_envs),  # terms
            np.zeros(self.envs.num_envs),  # truncs
            np.ones(self.envs.num_envs)  # is_first
        ]

    def _build_representation(self, representation_key: str,
                              input_space: Optional[gym.spaces.Space],
                              config: Optional[Namespace]) -> DreamerV3WorldModel:
        input_representations = dict(
            obs_shape=self.obs_shape,
            act_shape=self.act_shape,
            is_continuous=self.is_continuous,
            config=self.config,
        )
        representation = REGISTRY_Representation[representation_key](input_representations)
        if representation_key not in REGISTRY_Representation:
            raise AttributeError(f"{representation_key} is not registered in REGISTRY_Representation.")
        return representation

    def _build_memory(self, auxiliary_info_shape=None) -> SequentialReplayBuffer:
        input_buffer = dict(observation_space=self.observation_space,
                            action_space=self.action_space,
                            auxiliary_shape=auxiliary_info_shape,
                            n_envs=self.n_envs,
                            buffer_size=self.buffer_size,
                            batch_size=self.batch_size)
        return SequentialReplayBuffer(**input_buffer)

    def _build_policy(self) -> DreamerV3Policy:
        return REGISTRY_Policy["DreamerV3Policy"](self.models, self.config)

    def observe_and_action(self,
            obs: np.ndarray,
            deter: torch.Tensor,
            stoch: torch.Tensor,
            prev_acts: np.ndarray,
            is_first: np.ndarray,
            test_mode: Optional[bool] = False) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """Returns actions.
        Parameters:
            obs (np.ndarray): The observation.
            deter (torch.Tensor): The deterministic state of world model
            stoch (torch.Tensor): The stochastic state of world model
            prev_acts (np.ndarray): The previous actions (real_acts for envs.step())
            is_first (np.ndarray): is the obs the first obs of traj
            test_mode (Optional[bool]): True for testing without noises.
        Returns:
            actions: The real_actions to be executed.
        """
        if self.config.pixel:
            obs = obs.transpose(0, 3, 1, 2) / 255.0 - 0.5
        # -> [envs, *obs_shape]
        envs = obs.shape[0]
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        prev_acts = torch.as_tensor(prev_acts, device=self.device, dtype=torch.float32)
        is_first = torch.as_tensor(is_first, device=self.device, dtype=torch.float32)
        obs = obs.view(envs, *self.obs_shape)
        if not self.is_continuous:
            prev_acts = torch.nn.functional.one_hot(prev_acts.long(), num_classes=self.act_shape).float()
        prev_acts = prev_acts.view(envs, np.prod(self.act_shape))
        is_first = is_first.view(envs, 1)
        with torch.no_grad():
            with autocast(dtype=torch.float16):
                embed = self.models.encoder(obs)
                deter, stoch, _, _, _ = self.models.rssm.observe(deter, stoch, embed, prev_acts, is_first)
                latent = torch.cat([deter, stoch.view(*deter.shape[:-1], -1)], -1)
                acts, _ = self.policy.actor(latent, sample=not test_mode)
        # ont-hot -> real_actions
        if not self.is_continuous:
            acts = acts.argmax(dim=-1).detach().cpu().numpy()
        else:  # [envs, *act_shape]
            acts = acts.reshape(obs.shape[0], *self.act_shape).detach().cpu().numpy()
        """
        for env_interaction: actions.shape, (envs, ) or (env, *act_shape)
        """
        return acts, deter, stoch

    def train_epochs(self, n_epochs: int = 1):
        train_info = {}
        samples = self.memory.sample(self.config.seq_len)  # (envs, seq, batch, ~)
        if self.config.pixel:
            samples['obs'] = samples['obs'].transpose(0, 1, 2, 5, 3, 4) / 255.0 - 0.5
        # n_epoch(n_gradient step) scattered to each environment
        # st = np.random.choice(np.arange(self.envs.num_envs), 1).item()  # not necessary
        st = 0
        for _ in range(n_epochs):  # assert n_epochs == parallels
            cur_samples = {k: v[(st + _) % self.envs.num_envs] for k, v in samples.items()}
            train_info = self.learner.update(**cur_samples)
        return train_info

    def train(self, train_steps):  # each train still uses old rssm_states until episode end
        return_info = {}
        obs, acts, rews, terms, truncs, is_first = self.train_states

        for _ in tqdm(range(train_steps)):
            step_info = {}
            self.obs_rms.update(obs)
            obs = self._process_observation(obs)
            if self.current_step < self.start_training:  # ramdom_sample before training
                acts = np.array([self.envs.action_space.sample() for _ in range(self.envs.num_envs)])
            else:
                acts, self.deter, self.stoch = self.observe_and_action(obs, self.deter, self.stoch, acts, is_first)
            if self.atari:  # use truncs to train in xc_atari
                terms = deepcopy(truncs)
            """(o1, a1, r1, term1, trunc1, is_first1), acts: real_acts"""
            self.memory.store(obs, acts, self._process_reward(rews), terms, truncs, is_first)
            next_obs, rews, terms, truncs, infos = self.envs.step(acts)
            """
            set to zeros after the first step
            (o2, a1, r2, term2, trunc2, is_first2)
            """
            is_first = np.zeros_like(terms)
            obs = next_obs
            self.returns = self.gamma * self.returns + rews
            done_idxes = []
            for i in range(self.n_envs):
                if terms[i] or truncs[i]:
                    if self.atari and (~truncs[i]):  # do not term until trunc
                        pass
                    else:
                        # carry the reset procedure to the outside
                        done_idxes.append(i)
                        self.ret_rms.update(self.returns[i:i + 1])
                        self.returns[i] = 0.0
                        self.current_episode[i] += 1
                        if self.use_wandb:
                            step_info[f"Episode-Steps/rank_{self.rank}/env-{i}"] = infos[i]["episode_step"]
                            step_info[f"Train-Episode-Rewards/rank_{self.rank}/env-{i}"] = infos[i]["episode_score"]
                        else:
                            step_info[f"Episode-Steps/rank_{self.rank}"] = {f"env-{i}": infos[i]["episode_step"]}
                            step_info[f"Train-Episode-Rewards/rank_{self.rank}"] = {
                                f"env-{i}": infos[i]["episode_score"]}
                        self.log_infos(step_info, self.current_step)
                        return_info.update(step_info)
            self.current_step += self.n_envs
            if len(done_idxes) > 0:
                """
                store the last data and reset all
                (o_t, a_t = 0 for dones, r_t, term_t, trunc_t, is_first_t)
                """
                extra_shape = () if not self.is_continuous else self.act_shape
                acts[done_idxes] = np.zeros((len(done_idxes),) + extra_shape)
                if self.atari:  # use truncs to train in xc_atari
                    terms = deepcopy(truncs)
                self.memory.store(obs, acts, self._process_reward(rews), terms, truncs, is_first)

                """reset DreamerV3 Player's states"""
                obs[done_idxes] = np.stack([infos[idx]["reset_obs"] for idx in done_idxes])  # reset obs
                self.envs.buf_obs[done_idxes] = obs[done_idxes]
                rews[done_idxes] = np.zeros(len(done_idxes))
                terms[done_idxes] = np.zeros(len(done_idxes))
                truncs[done_idxes] = np.zeros(len(done_idxes))
                is_first[done_idxes] = np.ones_like(terms[done_idxes])

                self.deter[done_idxes] = torch.zeros(len(done_idxes), self.deter_size).to(self.config.device)
                self.stoch[done_idxes] = torch.zeros(len(done_idxes), self.stoch_size, self.classes).to(self.config.device)
            """
            start training 
            replay_ratio = self.gradient_step / self.current_step
            """
            if self.current_step > self.start_training:
                # count current_step when start_training
                n_epochs = max(int((self.current_step - self.start_training) * self.replay_ratio - self.gradient_step), 0)
                train_info = self.train_epochs(n_epochs=n_epochs)
                self.gradient_step += n_epochs
                if train_info is not None:
                    self.log_infos(train_info, self.current_step)
                    return_info.update(train_info)
        # save the train_states for next train
        self.train_states = [obs, acts, rews, terms, truncs, is_first]
        return return_info

    def test(self, env_fn, test_episodes: int) -> list:
        test_envs = env_fn()
        num_envs = test_envs.num_envs

        # init latent state
        deter = torch.zeros(num_envs, self.deter_size).to(self.config.device)
        stoch = torch.zeros(num_envs, self.stoch_size, self.classes).to(self.config.device)
        acts = np.zeros(num_envs, np.prod(self.act_shape))
        is_first = np.ones(num_envs)

        videos, episode_videos = [[] for _ in range(num_envs)], []
        current_episode, scores, best_score = 0, [], -np.inf
        obs, infos = test_envs.reset()
        if self.config.render_mode == "rgb_array" and self.render:
            images = test_envs.render(self.config.render_mode)
            for idx, img in enumerate(images):
                videos[idx].append(img)

        is_done = np.zeros(num_envs)
        while is_done.sum() < test_episodes:
            self.obs_rms.update(obs)
            obs = self._process_observation(obs)
            acts, deter, stoch = self.observe_and_action(obs, deter, stoch, acts, is_first, True)
            next_obs, rews, terms, truncs, infos = test_envs.step(acts)
            is_first = np.zeros_like(is_first)
            if self.config.render_mode == "rgb_array" and self.render:
                images = test_envs.render(self.config.render_mode)
                for idx, img in enumerate(images):
                    videos[idx].append(img)

            obs = deepcopy(next_obs)
            done_idxes = []
            for i in range(num_envs):
                if terms[i] or truncs[i]:
                    if self.atari and (~truncs[i]):
                        pass
                    else:
                        done_idxes.append(i)
                        obs[i] = infos[i]["reset_obs"]
                        if is_done[i] != 1:
                            is_done[i] = 1
                            scores.append(infos[i]["episode_score"])
                        if best_score < infos[i]["episode_score"]:
                            best_score = infos[i]["episode_score"]
                            episode_videos = videos[i].copy()
                        if self.config.test_mode:
                            print("Episode: %d, Score: %.2f" % (current_episode, infos[i]["episode_score"]))
            if len(done_idxes) > 0:
                deter[done_idxes] = torch.zeros(len(done_idxes), self.deter_size).to(self.config.device)
                stoch[done_idxes] = torch.zeros(len(done_idxes), self.stoch_size, self.classes).to(self.config.device)
                extra_shape = () if not self.is_continuous else self.act_shape
                acts[done_idxes] = np.zeros((len(done_idxes),) + extra_shape)
                is_first[done_idxes] = np.ones(len(done_idxes))

        if self.config.render_mode == "rgb_array" and self.render:
            # time, height, width, channel -> time, channel, height, width
            videos_info = {"Videos_Test": np.array([episode_videos], dtype=np.uint8).transpose((0, 1, 4, 2, 3))}
            self.log_videos(info=videos_info, fps=self.fps, x_index=self.current_step)  # fps cannot work

        if self.config.test_mode:
            print("Best Score: %.2f" % best_score)

        test_info = {
            "Test-Episode-Rewards/Mean-Score": np.mean(scores),
            "Test-Episode-Rewards/Std-Score": np.std(scores)
        }
        self.log_infos(test_info, self.current_step)

        test_envs.close()


        return scores


import argparse
import random
from collections import deque

import numpy as np
from copy import deepcopy

from gymnasium.spaces import Box, Discrete

from xuance.torch.utils.operations import set_seed
from xuance.common import get_configs, recursive_dict_update
from xuance.environment import make_envs, RawEnvironment, REGISTRY_ENV
from xuance.torch.agents import DreamerV3Agent

import cv2

class MyEnv(RawEnvironment):
    def __init__(self, config):
        super(MyEnv, self).__init__()
        # is_pixel & img_size
        self.pixel = config.pixel
        self.img_size = [64, 64]

        # frame stack
        self.frame_stack = config.frame_stack
        self.q = deque([], maxlen=self.frame_stack)
        # env_id
        self.env_id = config.env_id
        # obs_space & act_space
        self.obs_low, self.obs_high = -200, 200  # raw_obs range
        if not self.pixel:
            self.observation_space = Box(-200, 200, shape=[1, ], dtype=np.int16)
        else:
            self.observation_space = Box(
                low=0, high=255,
                shape=(self.img_size[0], self.img_size[1], 3 * self.frame_stack),
                dtype=np.uint8
            )
        self.action_space = Discrete(3)  # 0, 1, 2: -1, 0, 1

        # step count
        self.max_episode_steps = 500
        self._current_step = 0
        # set seeds
        self.action_space.seed(seed=config.env_seed)
        # scale & repeat
        self.scale = 4
        # obs_init (raw obs)
        self.o = self._obs_init()

    def _obs_init(self):  # TODO test
        # obs_init
        # o = np.array(0, dtype=np.int16)
        o = np.array(random.randint(-3, 3) * self.scale, dtype=np.int16)
        return o

    def _append_get_obs(self):
        if not self.pixel:
            return self.o
        # self.o -> deque, then get stacked obs from deque
        o = cv2.resize(self.render(), self.img_size, interpolation=cv2.INTER_AREA).astype(np.uint8)
        while len(self.q) < self.frame_stack:  # deal with first frame_stack
            self.q.append(o)
        self.q.append(o)  # then append new!!!
        return np.concatenate(list(self.q), axis=-1)

    def reset(self, **kwargs):
        self._current_step = 0
        # self.o = np.array(0, dtype=np.int16)
        self.o = self._obs_init()
        return self._append_get_obs(), {}

    def step(self, action):
        nxt_o = np.clip(self.o + (action - 1) * self.scale, self.obs_low, self.obs_high)
        # r = 1 if abs(nxt_o) == 0 else 0  # sparse reward!!!
        r = 1 if abs(nxt_o) < abs(self.o) else (-1 if abs(nxt_o) > abs(self.o) else 0)
        r *= self.scale
        # update obs
        self.o = nxt_o
        self._current_step += 1
        truncated = False if self._current_step < self.max_episode_steps else True
        terminated = truncated
        info = {}
        return self._append_get_obs(), r, terminated, truncated, info

    def render(self, *args, **kwargs):
        img = np.full([400, 400, 3], 0, dtype=np.uint8)  # uint8: [0, 2^8 - 1]
        # x: u -> d; y: l -> r
        sx, sy = 200, 200
        x, y = sx, sy + self.o
        for i in range(sx - 2, sx + 2):
            for j in range(sy - 2, sy + 2):
                img[i, j, 1] = 255
        for i in range(max(0, x - 10), min(399, x + 10)):
            for j in range(max(0, y - 10), min(399, y + 10)):
                img[i, j, 0] = 255
        return img

    def close(self):
        return


def parse_args():
    # MyEnv/myenv
    parser = argparse.ArgumentParser("Example of XuanCe: DreamerV3 for myenv.")
    parser.add_argument("--env-id", type=str, default="myenv")
    parser.add_argument("--log-dir", type=str, default="./logs/myenv/")
    parser.add_argument("--model-dir", type=str, default="./models/myenv/")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--harmony", type=bool, default=False)
    parser.add_argument("--pixel", type=bool, default=True)
    parser.add_argument("--frame-stack", type=int, default=1)

    parser.add_argument("--running-steps", type=int, default=10_000)  # 10k
    parser.add_argument("--eval-interval", type=int, default=200)  # 50 logs
    parser.add_argument("--replay-ratio", type=int, default=1)

    # parallels & benchmark
    parser.add_argument('--parallels', type=int, default=1)
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument("--benchmark", type=int, default=1)

    # render
    parser.add_argument('--render', type=str, default=True)
    parser.add_argument('--render-mode', type=str, default="rgb_array")

    # test
    # parser.add_argument('--test_episode', type=int, default=1)
    return parser.parse_args()


if __name__ == '__main__':
    parser = parse_args()
    configs_dict = get_configs(file_dir="config/myenv.yaml")
    configs_dict = recursive_dict_update(configs_dict, parser.__dict__)
    configs = argparse.Namespace(**configs_dict)

    REGISTRY_ENV[configs.env_name] = MyEnv
    set_seed(configs.seed)
    envs = make_envs(configs)
    Agent = DreamerV3Agent(config=configs, envs=envs)

    train_information = {"Deep learning toolbox": configs.dl_toolbox,
                         "Calculating device": configs.device,
                         "Algorithm": configs.agent,
                         "Environment": configs.env_name,
                         "Scenario": configs.env_id}
    for k, v in train_information.items():
        print(f"{k}: {v}")

    if configs.benchmark:
        def env_fn():
            configs_test = deepcopy(configs)
            configs_test.parallels = configs_test.test_episode
            return make_envs(configs_test)


        train_steps = configs.running_steps // configs.parallels
        eval_interval = configs.eval_interval // configs.parallels
        test_episode = configs.test_episode
        num_epoch = int(train_steps / eval_interval)

        test_scores = Agent.test(env_fn, test_episode)
        Agent.save_model(model_name="best_model.pth")
        best_scores_info = {"mean": np.mean(test_scores),
                            "std": np.std(test_scores),
                            "step": Agent.current_step}
        for i_epoch in range(num_epoch):
            print("Epoch: %d/%d:" % (i_epoch, num_epoch))
            Agent.train(eval_interval)
            test_scores = Agent.test(env_fn, test_episode)

            can_save = np.mean(test_scores) > best_scores_info["mean"]
            can_save |= (abs(np.mean(test_scores) - best_scores_info["mean"]) < 1e-6
                         and np.std(test_scores) < best_scores_info["std"])
            if can_save:
                best_scores_info = {"mean": np.mean(test_scores),
                                    "std": np.std(test_scores),
                                    "step": Agent.current_step}
                # save best model
                Agent.save_model(model_name="best_model.pth")
        # end benchmarking
        print("Best Model Score: %.2f, std=%.2f" % (best_scores_info["mean"], best_scores_info["std"]))
    else:
        if configs.test:
            def env_fn():
                configs.parallels = configs.test_episode
                return make_envs(configs)

            model = None
            # model = 'seed_1_2025_0324_100206'
            Agent.load_model(path=Agent.model_dir_load)
            scores = Agent.test(env_fn, configs.test_episode)
            print(f"Mean Score: {np.mean(scores)}, Std: {np.std(scores)}")
            print("Finish testing.")
        else:
            Agent.train(configs.running_steps // configs.parallels)
            Agent.save_model("final_train_model.pth")
            print("Finish training!")

    Agent.finish()

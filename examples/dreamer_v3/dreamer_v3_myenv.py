import argparse
import random

import numpy as np
from copy import deepcopy

from gymnasium.spaces import Discrete, Box

from xuance.torch.utils.operations import set_seed
from xuance.common import get_configs, recursive_dict_update
from xuance.environment import make_envs, RawEnvironment, REGISTRY_ENV
from xuance.torch.agents import DreamerV3Agent

class MyEnv(RawEnvironment):
    def __init__(self, config):
        super(MyEnv, self).__init__()
        self.env_id = config.env_id
        self.observation_space = Box(-200, 200, shape=[1, ], dtype=np.int16)
        self.action_space = Discrete(3)  # 0, 1, 2: -1, 0, 1
        self.max_episode_steps = 500
        self._current_step = 0
        # set seeds
        self.action_space.seed(seed=config.env_seed)
        # scale & repeat
        self.scale = 4
        # obs
        # self.o = np.array(0, dtype=np.int16)
        self.o = np.array(random.randint(-3, 3) * self.scale, dtype=np.int16)

    def reset(self, **kwargs):
        self._current_step = 0
        # self.o = np.array(0, dtype=np.int16)
        self.o = np.array(random.randint(-3, 3) * self.scale, dtype=np.int16)
        return self.o, {}
        # return self.observation_space.sample(), {}

    def step(self, action):
        nxt_o = np.clip(self.o + (action - 1) * self.scale,
                        self.observation_space.low[0], self.observation_space.high[0])
        r = 1 if abs(nxt_o) < abs(self.o) else (-1 if abs(nxt_o) > abs(self.o) else 0)
        r *= self.scale
        # update obs
        self.o = nxt_o

        self._current_step += 1
        truncated = False if self._current_step < self.max_episode_steps else True
        terminated = truncated
        info = {}
        return nxt_o, r, terminated, truncated, info

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
    parser = argparse.ArgumentParser("Example of XuanCe: DreamerV3 for MyEnv.")
    parser.add_argument("--env-id", type=str, default="myenv")
    parser.add_argument("--log-dir", type=str, default="./logs/myenv_old_v3/")
    parser.add_argument("--model-dir", type=str, default="./models/myenv_old_v3/")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--harmony", type=bool, default=False)

    # 10k
    """
    start_training: 64
    6.6it/s
    add encoder decoder
    seed_1_2025_0511_001530(1e-4, 8e-5 400step ok)
    add reward_predictor, discount_predictor
    seed_1_2025_0511_021155(1e-4,8e-5,ratio1 ok)
    seed_1_2025_0511_022653(1e-4,8e-5,ratio0.5 1.2k ok)
    remove old init!!!
    seed_1_2025_0511_131446(no_old_init x)
    seed_1_2025_0511_134516(no_old_rew_predictor_init x)
    seed_1_2025_0511_142503(rew_predictor outscale0.0 x, 1.0 ok)
    seed_1_2025_0511_143825(rew_predictor outscale 1.0 ok)
    """
    # TODO rssm
    parser.add_argument("--running-steps", type=int, default=10_000)  # 10k
    parser.add_argument("--eval-interval", type=int, default=200)  # 50 logs
    parser.add_argument("--replay-ratio", type=int, default=0.5)

    parser.add_argument("--render", type=bool, default=True)
    parser.add_argument("--render_mode", type=str, default="rgb_array")

    # parallels & benchmark
    parser.add_argument('--parallels', type=int, default=1)
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument("--benchmark", type=int, default=1)
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

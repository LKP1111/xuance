import argparse
import numpy as np
from copy import deepcopy
from xuance.torch.utils.operations import set_seed
from xuance.common import get_configs, recursive_dict_update
from xuance.environment import make_envs
from xuance.torch.agents import DreamerV3Agent

def parse_args():
    parser = argparse.ArgumentParser("Example of XuanCe: DreamerV3 for Atari.")
    parser.add_argument("--env-id", type=str, default="ALE/Breakout-v5")
    parser.add_argument("--log-dir", type=str, default="./logs/Breakout-v5/")
    parser.add_argument("--model-dir", type=str, default="./models/Breakout-v5/")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--harmony", type=bool, default=True)

    """
    S: 3460MiB
    Total params: 15,684,737
      encoder          689,600
      rssm             5,777,920
      observation_model 6,984,641
      reward_model     1,181,439
      continue_model   1,051,137
    actor Total params: 1,052,676
      model            1,050,624
      mlp_heads        2,052
    critic Total params: 1,181,439
      _model           1,181,439
    """

    # atari100k, ratio=1, gradient_step=100k
    parser.add_argument("--running-steps", type=int, default=100_000)  # 100k
    parser.add_argument("--eval-interval", type=int, default=2_000)  # 50 logs
    parser.add_argument("--replay-ratio", type=int, default=1)

    # parallels & benchmark
    parser.add_argument('--parallels', type=int, default=1)
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument("--benchmark", type=int, default=1)
    return parser.parse_args()

import torch
from collections import defaultdict
def count_parameters(model: torch.nn.Module):
    param_counts = defaultdict(int)
    for name, param in model.named_parameters():
        # name like 'encoder.conv1.weight' -> top_name='encoder'
        top_name = name.split('.')[0]
        param_counts[top_name] += param.numel()
    total = sum(param_counts.values())
    print(f"Total params: {total:,}")
    for module_name, cnt in param_counts.items():
        print(f"  {module_name:<16} {cnt:,}")

if __name__ == '__main__':
    # print(sys.path)  # python path
    parser = parse_args()
    configs_dict = get_configs(file_dir="config/atari.yaml")
    configs_dict = recursive_dict_update(configs_dict, parser.__dict__)
    configs = argparse.Namespace(**configs_dict)

    set_seed(configs.seed)
    envs = make_envs(configs)
    Agent = DreamerV3Agent(config=configs, envs=envs)

    count_parameters(Agent.policy.world_model)
    count_parameters(Agent.policy.actor)
    count_parameters(Agent.policy.critic)

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
            Agent.load_model(path=Agent.model_dir_load, model=model)
            scores = Agent.test(env_fn, configs.test_episode)
            print(f'scores: {scores}')
            print(f"Mean Score: {np.mean(scores)}, Std: {np.std(scores)}")
            print("Finish testing.")
        else:
            Agent.train(configs.running_steps // configs.parallels)
            Agent.save_model("final_train_model.pth")
            print("Finish training!")

    Agent.finish()

import argparse
import numpy as np
from copy import deepcopy
from xuance.torch.utils.operations import set_seed
from xuance.common import get_configs, recursive_dict_update
from xuance.environment import make_envs
from xuance.torch.agents import DreamerV3Agent

def parse_args():
    parser = argparse.ArgumentParser("Example of XuanCe: DreamerV3 for Atari.")
    parser.add_argument("--env-id", type=str, default="ALE/Boxing-v5")
    parser.add_argument("--log-dir", type=str, default="./logs/Boxing-v5/")
    parser.add_argument("--model-dir", type=str, default="./models/Boxing-v5/")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--harmony", type=bool, default=False)

    # atari100k, ratio=0.25, gradient_step=25k; config: 200m;
    # Total params: 176,680,966;
    #   no amp, f32: 22874MB, 4.88it/s
    #   amp, bf16: 20748MiB, 4.3it/s;
    #   amp, gradscaler, f16: 19854MiB, 5.4it/s
    # after compile
    #   no amp, f32: 23224MiB; : 5.8it/s
    #   amp, bf16: 21074MiB, 3.45it/s;
    #   amp, f16: 23492MiB, f16, 6.8it/s
    """
    100m in sc_lkp, f32
    Total params: 99,551,430; 
    250W; 15539MiB; 70~97%; 5.2it/s
    
    100m in sc, f32, torch 1.13.0
    Total params: 99,548,356;
    287~307W; 19339MiB; 50~90%, 5.8it/s

    200m in sc, f32, torch 1.13.0
    Total params: 176,676,868
    310~340W; 22943MiB; 57~100%, 4.62it/s
    
    Boxing 
    seed_1_2025_0512_053525(-> 0 x)
    
    myenv
    TODO
    """
    parser.add_argument("--running-steps", type=int, default=100_000)  # 100k
    parser.add_argument("--eval-interval", type=int, default=2_000)  # 50 logs
    parser.add_argument("--replay-ratio", type=int, default=0.25)

    # render
    parser.add_argument('--render', type=str, default=True)
    parser.add_argument('--render-mode', type=str, default="rgb_array")

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
    # import sys
    # print(sys.path)  # python path
    # sys.path.append('/home/lkp/projects/xc_official')
    
    parser = parse_args()
    # configs_dict = get_configs(file_dir="config/atari.yaml")
    configs_dict = get_configs(file_dir="config/atari(100m).yaml")
    configs_dict = recursive_dict_update(configs_dict, parser.__dict__)
    configs = argparse.Namespace(**configs_dict)

    set_seed(configs.seed)
    envs = make_envs(configs)
    Agent = DreamerV3Agent(config=configs, envs=envs)
    
    count_parameters(Agent.models)

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

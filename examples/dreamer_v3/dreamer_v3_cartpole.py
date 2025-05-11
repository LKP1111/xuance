import argparse
import numpy as np
from copy import deepcopy
from xuance.torch.utils.operations import set_seed
from xuance.common import get_configs, recursive_dict_update
from xuance.environment import make_envs
from xuance.torch.agents import DreamerV3Agent

def parse_args():
    parser = argparse.ArgumentParser("Example of XuanCe: DreamerV3 for CartPole.")
    parser.add_argument("--env-id", type=str, default="CartPole-v1")
    parser.add_argument("--log-dir", type=str, default="./logs/CartPole-v1/")
    parser.add_argument("--model-dir", type=str, default="./models/CartPole-v1/")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--harmony", type=bool, default=False)

    """
    12m, f16, 4596MB
    Total params: 10,446,084
    seed_1_2025_0506_234045 (12m,4e-5,ratio0.5) x
    seed_1_2025_0507_021601 (12m,4e-3,4e-5,ratio0.5) x
    seed_1_2025_0507_021632 (12m,4e-4,4e-5,ratio0.5) x
    seed_1_2025_0507_231944 (12m,4e-3,1e-5,ratio0.5) x
    seed_1_2025_0507_232132 (12m,4e-3,4e-6,ratio0.5) x
    f32, 5144MiB, 8.9it/s
    seed_1_2025_0508_230457(12m,4e-3,4e-4,ratio0.5,f32) x
    seed_1_2025_0508_230648(12m,4e-3,4e-5,ratio0.5,f32) (<10k,b1;>10k x)
    seed_1_2025_0509_001710(12m,4e-5,ratio0.5,f32) x
    seed_1_2025_0509_003415(12m,4e-3,1e-5,ratio0.5,f32) (500) x
    seed_1_2025_0509_003604(12m,4e-3,8e-5,ratio0.5,f32) x
    seed_1_2025_0509_011346(12m,4e-3,2e-5,ratio0.5,f32) x
    seed_1_2025_0509_011607(12m,1e-3,4e-5,ratio0.5,f32) x
    (12m,4e-3,4e-5,ratio0.5,f32,kl_dyn0.5,cont_layer3) x
    (12m,1e-3,4e-5,ratio0.5,f32,kl_dyn0.5,cont_layer3) x
    seed_1_2025_0510_191214(12m,4e-3,4e-5,ratio1,f32) x
    ---
    seed_1_2025_0511_172747(1m,4e-5,ratio1,f32) 6k,10k->500 (ok)
    
    Total params: 695,172
    1m, f32, 10.5it/s, 1420MiB
    seed_1_2025_0509_022723(1m,4e-3,4e-5,ratio0.5,f32) (500) x
    seed_1_2025_0509_022751(1m,4e-3,4e-4,ratio0.5,f32) (500) x
    seed_1_2025_0511_153620(1m,4e-5,ratio0.5,f32 rew_pred outscale 0.0 -> 1.0 (500) x)
    seed_1_2025_0511_164705(1m,4e-5,ratio0.5,f32, critic outscale -> 1.0, x)
    ---
    seed_1_2025_0511_165928(1m,4e-5,ratio0.5,f32, actor outscale -> 1.0) 50k -> 500 (nearly ok)
    seed_1_2025_0511_172540(1m,4e-5,ratio1,f32) (20k -> 500) (ok)
    
    50m, f32, 8956MiB
    seed_1_2025_0510_192244(50m,4e-3,4e-5,ratio0.5,f32) x
    
    
    """
    parser.add_argument("--running-steps", type=int, default=50_000)  # 10k
    parser.add_argument("--eval-interval", type=int, default=200)  # 50 logs
    parser.add_argument("--replay-ratio", type=int, default=1)

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
    parser = parse_args()
    # configs_dict = get_configs(file_dir="config/CartPole-v1.yaml")
    configs_dict = get_configs(file_dir="config/CartPole-v1(12m).yaml")
    # configs_dict = get_configs(file_dir="config/CartPole-v1(50m).yaml")
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
            Agent.load_model(path=Agent.model_dir_load)
            scores = Agent.test(env_fn, configs.test_episode)
            print(f"Mean Score: {np.mean(scores)}, Std: {np.std(scores)}")
            print("Finish testing.")
        else:
            Agent.train(configs.running_steps // configs.parallels)
            Agent.save_model("final_train_model.pth")
            print("Finish training!")

    Agent.finish()

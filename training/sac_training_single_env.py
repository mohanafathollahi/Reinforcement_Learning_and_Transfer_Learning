import os
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from custom_env.base_custom_env import CustomEnv
from config.env_config import env_config
from utils.protocol_choosing_source_model import choose_source_model
from utils.transfer_learning_helper import get_reward_shaping_model
import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu_number', default=0)
    parser.add_argument('-t', '--timesteps', default=100000)
    parser.add_argument('-d', '--cuda_devices', default="0")
    args = parser.parse_args()
    print(f"ARGS: {args}")

    cuda_devices = str(args.cuda_devices)
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices

    gpu_number = str(args.gpu_number)
    # device = f"cuda:{gpu_number}" #activate it when u are running in gpu
    device = "cpu"              #activate it when u are running in cpu
    print(device)
    timesteps = int(args.timesteps)

    for env in env_config[:1]: #first and only env
        volt_lb = env.get("volt_lb")
        volt_up = env.get("volt_up")
        pref_lb = env.get("pref_lb")
        pref_ub = env.get("pref_ub")
        step = env.get("pref_step")
        penalty_coefficient = env.get("penalty_coefficient")
        pref_range = list(range(pref_lb, pref_ub, step))
        log_suffix = f"volt_{volt_lb}_{volt_up}_pref_{pref_ub - step}"
        tensorboard_log = f"../logs/tensorboard_transfer_{log_suffix}/"
        save_path = f"../logs/policy_transfer_{log_suffix}/"

        env = CustomEnv(volt_lb=volt_lb, 
                        volt_up=volt_up, 
                        penalty_coefficient=penalty_coefficient,
                        pref_range=pref_range)

        checkpoint_callback = CheckpointCallback(
            save_freq=500,
            save_path=save_path,
            save_replay_buffer=False,
            save_vecnormalize=False,
            name_prefix="sac_model_transfer"
        )
        #select model
        _,info = env.reset()
        vga, vgb, vgc, pref = info['vga'], info['vgb'], info['vgc'], info['pref']
        print(f"info:{info}")

        #Choose the closest source model to the taget observation
        desire_source_model, desire_env_id = choose_source_model(vga, vgb, vgc, pref, device)
        
        print(type(desire_source_model))
        devices = set(p.device for p in desire_source_model.policy.parameters())
        print(devices)
        
        target_env =  CustomEnv(volt_lb = volt_lb,
                                volt_up = volt_up,
                                penalty_coefficient = penalty_coefficient,
                                pref_range = list(range(pref_lb, pref_ub, step)))

        target_env_monitor_rs = Monitor(target_env, '../logs/transfer/')
        target_reward_reshaping_model = get_reward_shaping_model(policy_name = 'MlpPolicy',
                                                                 env = target_env_monitor_rs,
                                                                 src_model = desire_source_model,
                                                                 learning_rate = 3e-4,
                                                                 tensorboard_log = tensorboard_log,
                                                                 use_sde = True,
                                                                 device = device,
                                                                 verbose = 1,
                                                                 desire_env_id = desire_env_id,
                                                                 pref = pref,
                                                                 )

        weights = target_reward_reshaping_model.policy.state_dict()
        for key, value in weights.items():
            print(key, len(value))
        # print(f"show the device to train the env: {target_reward_reshaping_model.device}")
        # target_reward_reshaping_model.learn(total_timesteps=1000000,
        #                                     log_interval=1,
        #                                     callback=checkpoint_callback
        #                                     )


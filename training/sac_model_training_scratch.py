import argparse
from config.env_config import env_config
from stable_baselines3.common.callbacks import CheckpointCallback
from custom_env.clipaction_env import CustomEnv
from models.sac_model import SacModel
from stable_baselines3.common.monitor import Monitor

import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu_number', default=0)
    parser.add_argument('-t', '--timesteps', default = 2_000_000)
    parser.add_argument('-el', '--episode_length', default = 1)
    parser.add_argument('-s', '--seed',default = 0, type =int)
    parser.add_argument('-n', '--model_name', default = "episode_1_MapAction" ,type =str)

#     parser.add_argument('-d', '--cuda_devices', default="0")
    args = parser.parse_args()

    gpu_number = str(args.gpu_number)
    print(f"gpu_number:{gpu_number}")
    if int(gpu_number) >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_number)
        device = f"cuda:{gpu_number}"
    else:
        device = "cpu"

    timesteps = int(args.timesteps)

    for env in env_config[:1]:  # first and only env
        volt_lb = env.get("volt_lb")
        volt_up = env.get("volt_up")
        pref_lb = env.get("pref_lb")
        pref_ub = env.get("pref_ub")
        step = env.get("pref_step")

        pref_range = list(range(pref_lb, pref_ub, step))

        tensorboard_log = f"../logs_{args.model_name}/tensorboard/"
        save_path = f"../logs_{args.model_name}/policy/"

        env = CustomEnv(volt_lb=volt_lb,
                        volt_up=volt_up,
                        episode_length=args.episode_length,
                        pref_range= pref_range,
                        )
        env = Monitor(env, f"../logs_{args.model_name}/monitor/")
        checkpoint_callback = CheckpointCallback(save_freq=2000,
                                                save_path=save_path,
                                                save_replay_buffer=False,
                                                save_vecnormalize=False
                                                )
        # select model

        train_sac_model = SacModel()
        print()
        try:
            train_sac_model.train_sac_model(env = env,
                                            device=device,
                                            timesteps=timesteps,
                                            checkpoint_callback=checkpoint_callback,
                                            tensorboard_log=tensorboard_log)

        except KeyboardInterrupt as e:
            print("Finishing Training...")
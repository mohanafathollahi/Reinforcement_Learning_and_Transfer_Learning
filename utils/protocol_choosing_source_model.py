import os
from config.env_config_envs import env_configs
from models.sac_model import SacModel
from custom_env.base_custom_env import CustomEnv



def choose_source_model(vga, vgb, vgc, pref, device):
    voltage_mean = (vga + vgb + vgc) / 3
    # desire_env = 0
    for env in env_configs:
        if (
            pref <= env.get("pref_ub")
            and pref >= env.get("pref_lb")
            and voltage_mean <= env.get("volt_up")
            and voltage_mean >= env.get("volt_lb")
        ):
            # env_id  = env.get("env_id")
            desire_env = env
            desire_env_id = env['env_id']
            # print(f"env_id:{desire_env_id}")
            break
    main_pretrain_model = f"./blob_models/pretrained_model/env_{env['env_id']}"
    pretrain_model = os.listdir(main_pretrain_model)[0]
    pretrain_model_path = os.path.join(main_pretrain_model, pretrain_model)

    env = CustomEnv(volt_lb=desire_env['volt_lb'],
                    volt_up=desire_env['volt_up'],
                    penalty_coefficient=desire_env['penalty_coefficient'],
                    pref_range=list(range(desire_env['pref_lb'], desire_env['pref_ub'], desire_env['pref_step'])))

    appropriate_source_model = SacModel()
    desire_source_model = appropriate_source_model.get_pretrained_output(env=env,
                                                           device=device,
                                                           directory_pretrain_model = pretrain_model_path,
                                                           return_model_only=True)

    return desire_source_model, desire_env_id



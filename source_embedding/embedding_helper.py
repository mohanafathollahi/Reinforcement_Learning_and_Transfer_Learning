import torch as th
from models.sac_model import SacModel
from custom_env.base_custom_env import CustomEnv
from config.env_config_envs import env_configs
import pandas as pd
import os
from utils.penultimate import penultimate

def create_random_samples(sample_df_path = "./testing/csv_files/filled_by_rl_and_penalty_added.csv",
                          number_of_samples: int = 10) -> pd.DataFrame:
    sample_df = pd.read_csv(sample_df_path)
    group_cols = ["env_id", "pref"]
    target_cols = ["env_id", "pref", "vga", "vgb", "vgc", "RL_Iqa", "RL_Iqb", "RL_Iqc"]
    sampled_df = sample_df.groupby(group_cols)[target_cols].sample(number_of_samples, random_state=43)
    sampled_df = sampled_df.reset_index(drop=True)
    return sampled_df

def path_pretrained_model(env_id: str,path_pretrained_model: str = "./blob_models/pretrained_model") -> str:
    pure_id = env_id.split("_")[1]
    subset_path = os.path.join(path_pretrained_model,f"env_{pure_id}")
    model_path = os.listdir(subset_path)[0]
    final_path = os.path.join(subset_path, model_path)
    return final_path

def upload_pretrain_model(env_id, vga, vgb, vgc, pref):
    env = CustomEnv()
    for various_env in env_configs:
        if various_env["env_id"] == int(env_id.split("_")[1]):
            action, p, model = SacModel().get_pretrained_output(env, device="auto",
                                                                    vga = vga,
                                                                    vgb = vgb,
                                                                    vgc = vgc,
                                                                    pref = pref,
                                                                    deterministic = True,
                                                                    directory_pretrain_model = path_pretrained_model(env_id),
                                                                    return_model_only = False)
            return model, action
    return None

def create_source_embedding_tensor(sample_df: pd.DataFrame, save_path = "./source_embedding/embedding_sources/",
                            v_nominal = 110, iq_nominal= 10, p_nominal= 2000) -> th.Tensor:
    sample_df[["penultimate_qf0", "q_value_qf0", "penultimate_qf1", "q_value_qf1"]] = None
    pen_qf0_dict, q_val_qf0_dict, pen_min_qf0_qf1_dict = {}, {}, {}
    pen_qf1_dict, q_val_qf1_dict, q_value_min_qf0_qf1_dict = {}, {}, {}
    for index, row in sample_df.iterrows(): #iterate over sample_df
        pen_qf0, q_val_qf0, pen_qf1, q_val_qf1, pen_min_qf0_qf1, q_value_min_qf0_qf1 = penultimate(
                        th.tensor([row["vga"] / v_nominal, row["vgb"] / v_nominal, row["vgc"] / v_nominal, row["pref"] / p_nominal],
                                  dtype=th.float32).unsqueeze(0),
                        th.tensor([row["RL_Iqa"] / iq_nominal, row["RL_Iqb"] / iq_nominal, row["RL_Iqc"] / iq_nominal],
                                  dtype=th.float32).unsqueeze(0),
                        upload_pretrain_model(row["env_id"], row["vga"], row["vgb"], row["vgc"], row["pref"])[0]
        )

        key = (row["env_id"], row["pref"])
        for store, tensor in [(pen_qf0_dict, pen_qf0),
                              (q_val_qf0_dict, q_val_qf0),
                              (pen_qf1_dict, pen_qf1),
                              (q_val_qf1_dict, q_val_qf1),
                              (pen_min_qf0_qf1_dict, pen_min_qf0_qf1),
                              (q_value_min_qf0_qf1_dict, q_value_min_qf0_qf1)]:
            if key not in store:
                store[key] = tensor
            else:
                store[key] = th.vstack([store[key], tensor])

    # print(q_val_qf0, q_val_qf1,q_value_min_qf0_qf1)
    if save_path:
        th.save(pen_qf0_dict, f"{save_path}penultimate_criticNet0.pt")
        th.save(q_val_qf0_dict, f"{save_path}q_val_criticNet0.pt")

        th.save(pen_qf1_dict, f"{save_path}penultimate_criticNet1.pt")
        th.save(q_val_qf1_dict, f"{save_path}q_val_criticNet1.pt")

        th.save(pen_min_qf0_qf1_dict, f"{save_path}penultimate_min_criticNets.pt")
        th.save(q_value_min_qf0_qf1_dict, f"{save_path}q_val_min_criticNets.pt")


        print(f"Saved embeddings to '{save_path}*.pt'")
    #
    # return pen_qf0_dict, q_val_qf0_dict, pen_qf1_dict, q_val_qf1_dict
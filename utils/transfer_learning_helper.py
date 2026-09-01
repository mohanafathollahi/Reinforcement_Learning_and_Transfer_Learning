from utils.reward_shapers import RewardShaperGeneral
from utils.customSAC_byRewardShaped import CustomSACbyRewardShaped
from utils.customTD3_byRewardShaped import CustomTD3byRewardShaped

from source_embedding.get_source_embeddings import get_source_embeddings


def get_reward_shaping_model(policy_name,
                             env, 
                             src_model,
                             learning_rate,
                             tensorboard_log,
                             use_sde, 
                             device,
                             verbose,
                             pref,
                             desire_env_id,
                             embedding_type,
                             model_name,
                             stats_window_size,
                             **kwargs,
                             ):
    
    reward_shaper = create_reward_shaper(src_model,
                                          desire_env_id,
                                          pref,
                                          device,
                                          embedding_type)
    #it is the part that conncet to changed implementation of SAC model.
    if model_name == "SAC":
        target_model_reshape = CustomSACbyRewardShaped(policy = policy_name,
                                                       env = env,
                                                       learning_rate = learning_rate,
                                                       tensorboard_log = tensorboard_log,
                                                       use_sde = use_sde,
                                                       device = device,
                                                       verbose=verbose,
                                                       reward_shaper=reward_shaper,
                                                       **kwargs)
    elif model_name == "TD3":
        print(f"model_name is:{model_name}")
        target_model_reshape = CustomTD3byRewardShaped(policy=policy_name,
                                                       env=env,
                                                       learning_rate=learning_rate,
                                                       tensorboard_log=tensorboard_log,
                                                       device=device,
                                                       verbose=verbose,
                                                       reward_shaper=reward_shaper,
                                                       stats_window_size= stats_window_size,
                                                       **kwargs)
    return target_model_reshape

def create_reward_shaper(source_model,
                          desire_env_id,
                          pref,
                          device,
                          embedding_type):
    # torch.Size([10, 256])--> (batch_size, embedding_size)
    #the embedding is for penultimate of first critic network.
    embeddings, q_vals = get_source_embeddings(desire_env_id,
                                               pref,
                                               embedding_type)

    # print(f"type of  embeeding: {type(embeddings)}, type of q_vals: {type(q_vals)}")

    reward_shaper = RewardShaperGeneral(source_model,
                                         embeddings,
                                         q_vals,
                                         source_model.gamma,
                                         device,
                                         embedding_type)

    return reward_shaper
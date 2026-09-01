import random
import platform
import onnxruntime as ort
import numpy as np
from stable_baselines3.common.callbacks import CheckpointCallback
import os




def penalize_invalid_action(voltage,
                            generated_action,
                            penalty_coefficient):
    #generated_action and lowerbound should belong to [0,1]
    # print(f"voltage: {voltage}, generated_action: {generated_action}, MRC:{lowerbound(voltage)},penalty_coefficient: {penalty_coefficient}")
    if generated_action < lowerbound(voltage):
        penalty = penalty_coefficient*1
    else:
        penalty = 0
    return penalty

def reward_func(v_inverter, voltage_coefficient:int = 1):
    return voltage_coefficient*(v_inverter-1)**2

def norm_reward_calc(va_updated, vb_updated, vc_updated, p, scale_factor,p_coef= 0.01):
    reward = -1*((1-va_updated/scale_factor)**2 +
                 (1-vb_updated/scale_factor)**2 +
                 (1-vc_updated/scale_factor)**2 -
                  # p_coef*(p/scale_factor-pref/scale_factor)**2 -
                  p_coef*(p/scale_factor)**2)
    return reward

def lowerbound(v): #normalized v should pass to this func
    lower_bound = 0.9-1.4*(v-0.4)
    return lower_bound

def generate_random_voltage(low: int = None, up: int = None):
    a = random.uniform(low, up)
    b = random.uniform(low, up)
    c = random.uniform(low, up)
    all_vals = (a, b, c)
    min_val = min(a, b, c)
    max_val = max(a, b, c)
    middle_index = 3 - (all_vals.index(min_val) + all_vals.index(max_val))
    middle_val = all_vals[middle_index]
    if min_val > max_val - middle_val and abs(middle_val - min_val) > 5 and abs(middle_val - max_val) > 5:
        return all_vals
    else:
        return generate_random_voltage(low, up)  # Recursive call

def is_os(os_name: str):
    system_os = platform.system().lower()
    return os_name.lower() == system_os.lower()

def is_os_linux():
    return is_os('linux')

def is_os_windows():
    return is_os('windows')

def map_action(action, v_g): #normalized action and v_g
    L = lowerbound(v_g)
    # print(f"lowerbound:{L}")
    Iq = np.maximum(action, L)
    return Iq.astype(np.float32)

def unscale_action(low_action,
                   high_action,
                   action) -> np.ndarray:
    """
    Rescale the action from [-1, 1] to [low, high]
    (no need for symmetric action space)
    :param scaled_action: Action to un-scale
    """
    # assert isinstance(action_space, spaces.Box
    # ), f"Trying to unscale an action using an action space that is not a Box(): {self.action_space}"
    low, high = low_action, high_action
    return low + (0.5 * (action + 1.0) * (high - low))

def predict_iq_by_onnx(state, onnx_model):
        normalized_state = [state[i]/110 if i<=2 else state[i]/1400 for i in range(len(state))]
        # print(f"normalized_state:{normalized_state}")
        observation_np = np.array(normalized_state, dtype=np.float32).reshape(1, -1)
        ort_sess = ort.InferenceSession(onnx_model)
        action = ort_sess.run(None, {"input": observation_np})
        # print(f"action:{action}")
        scaled_action = unscale_action(low_action=0.25, high_action=1, action=action[0])
        # print(f"scaled_action:{scaled_action}")
        map_actions = map_action(np.array(scaled_action), np.array(normalized_state[:3]))
        valid_actions = np.array([a*10 for a in map_actions])
        return valid_actions

def checkpoint_callback(model_name: str,
                        env_name: str,
                        save_freq: int = 1000,
                        name_prefix: str = "sac_model_"):
    
    save_path = f"../logs/{model_name}_{env_name}/"
    return CheckpointCallback(save_freq = save_freq,
                                save_path = save_path,
                                save_replay_buffer = False,
                                save_vecnormalize = False,
                                name_prefix = name_prefix)
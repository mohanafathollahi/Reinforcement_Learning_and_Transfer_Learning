from helper.generic_functions import *
from sim_wrapper.simulator_wrapper import SimulatorWrapper
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import List


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface."""
    NOMINAL_VOLTAGE = 110
    MAX_ACTION = 10
    SCALE_PREF = 1400
    def __init__(self, volt_lb: int,
                 volt_up: int,
                 episode_length: int,
                 pref_range: list,
                 ):
        super().__init__()
        self.volt_lb = volt_lb
        self.volt_up = volt_up
        self.episode_length = episode_length
        self.pref_range = pref_range
        self.simulatorWrapper = SimulatorWrapper()

        low_ac_bound = np.array([0.25, 0.25, 0.25],  dtype=np.float32)  # lower bound of action scpace
        high_ac_bound = np.array([1, 1, 1],  dtype=np.float32)  # higher bound of action scpace
        self.action_space = spaces.Box(low=low_ac_bound, high=high_ac_bound, dtype=np.float32)

        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32),
            dtype=np.float32
        )

        self.i_scale_factor = self.MAX_ACTION
        self.v_scale_factor = self.NOMINAL_VOLTAGE  # normalize v
        self.num_steps = 0  # number of steps that we can boost the voltage

        self.vga = 0
        self.vgb = 0
        self.vgc = 0
        self.number_of_episodes = 0

    def reset(self, seed=None, options=None):
        self.pref = self.pref_range[self.number_of_episodes % len(self.pref_range)]
        self.num_steps = 0
        vga, vgb, vgc = generate_random_voltage(self.volt_lb, self.volt_up)

        self.vga = vga
        self.vgb = vgb
        self.vgc = vgc

        inputs = np.array([0.0, 0.0, 0.0, self.vga, self.vgb, self.vgc, self.pref],
                          dtype=np.float64)  # input is based on the reset condition
        outputs = np.zeros(20, dtype=np.float64)
        self.simulatorWrapper.run_simulation(inputs, outputs)

        self.va_rms_no_iq = outputs[3] / self.v_scale_factor
        self.vb_rms_no_iq = outputs[4] / self.v_scale_factor
        self.vc_rms_no_iq = outputs[5] / self.v_scale_factor

        observation = np.array([self.va_rms_no_iq, self.vb_rms_no_iq, self.vc_rms_no_iq,
                                self.pref / self.SCALE_PREF])
        # print(f"observation_in_reset:{observation}\n")
        info = {"vga": self.vga, "vgb": self.vgb, "vgc": self.vgc, "pref": self.pref}
        # print(f"info:{info}")
        return observation, info

    def step(self, action):

        action1 = action[0] * self.i_scale_factor
        action2 = action[1] * self.i_scale_factor
        action3 = action[2] * self.i_scale_factor
        # print(f"actions in custom env:{[action1, action2, action3]}")
        inputs = np.array([action1, action2, action3, self.vga, self.vgb, self.vgc, self.pref],
                          dtype=np.float64)
        # print(f"inputs:{inputs}")
        outputs = np.zeros(20, dtype=np.float64)
        self.simulatorWrapper.run_simulation(inputs, outputs)

        pref = self.pref / self.v_scale_factor

        boosted_voltage_a = outputs[3] / self.v_scale_factor
        boosted_voltage_b = outputs[4] / self.v_scale_factor
        boosted_voltage_c = outputs[5] / self.v_scale_factor
        boosted_p = outputs[15] / self.v_scale_factor
        # print(f"pref:{pref}, boosted_p:{boosted_p*self.v_scale_factor}")




        reward = (norm_reward_calc(va_updated=boosted_voltage_a,
                                  vb_updated=boosted_voltage_b,
                                  vc_updated=boosted_voltage_c,
                                  p=boosted_p, pref=pref, scale_factor=1, p_coef=0.01))



        # print(f"reward:{reward}")

        # observation = np.array([self.vga, self.vgb, self.vgc, self.pref])
        # What if the observation is based on applied action over the obs
        # observation = np.array([boosted_voltage_a, boosted_voltage_b, boosted_voltage_c, boosted_p])

        observation = np.array([self.va_rms_no_iq, self.vb_rms_no_iq,
                                self.vc_rms_no_iq, self.pref / self.SCALE_PREF])

        # print(f"observation:{observation}\n")

        self.num_steps += 1

        terminated = (self.num_steps == self.episode_length)
        if terminated:
            # print("\n")
            # print("****episode has been finished****\n")
            self.number_of_episodes += 1

        truncated = False
        info = {'reward': reward}
        return observation, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass

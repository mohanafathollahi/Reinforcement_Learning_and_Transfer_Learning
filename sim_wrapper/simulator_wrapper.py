import numpy as np
from helper.generic_functions import is_os_linux

if is_os_linux():
    import sys
    sys.path.append("sim_wrapper")
    import simulation

class SimulatorWrapper:
    def __init__(self):
        pass

    def output_simulation(self, action1, action2, action3, vga, vgb, vgc, pref):
        inputs = np.array([action1, action2, action3, vga, vgb, vgc, pref], dtype=np.float64)
        outputs = np.zeros(20, dtype=np.float64)
        outputs = self.run_simulation(inputs, outputs)
        return outputs

    def run_simulation(self, inputs, outputs):
        if is_os_linux():
            simulation.run_simulation(inputs, outputs)
        else:
            # Simulator not defined
            outputs = np.zeros(20, dtype=np.float64)

        return outputs


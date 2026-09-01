import numpy as np
import simulation
v = [77.91144,	58.15145,	52.04442]
inputs = np.array([5.2,	4.8,	3.7, 77.91144,	58.15145,	52.04442, 100 ], dtype=np.float64)
outputs = np.zeros(20, dtype=np.float64)
simulation.run_simulation(inputs, outputs)
Ipa= outputs[9]
Ipb = outputs[10]
Ipc = outputs[11]
va_p = outputs[3]
vb_p= outputs[4]
vc_p = outputs[5]
print(f"v_updated= {va_p, vb_p, vc_p}")

p = va_p*Ipa + vb_p*Ipb + vc_p*Ipc

print(f"p:{p/2}")

print('\n'.join([str(e) for e in list(outputs)]))


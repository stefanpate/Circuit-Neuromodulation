# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 16:14:10 2018

Simulation of a single neuron circuit.

@author: Luka
"""

import matplotlib.pyplot as plt
import numpy as np

from model import LocalizedConductance, Circuit

# Define current elements
i1 = LocalizedConductance(-2, 0, 'fast') # fast negative conductance
i2 = LocalizedConductance(2, 0, 'slow') # slow positive conductance
i3 = LocalizedConductance(-1.5, -1.5, 'slow') # slow negative conductance
i4 = LocalizedConductance(1.5, -1.5, 'ultraslow') # ultraslow positive conductance

# Interconnect the elements
circ = Circuit(i1, i2, i3, i4)

# Simulate the circuit
trange = (0, 10000)
v0 = (0, 0, 0.2, 0.3)
i_app = lambda t: -2 # define i_app as function of t
sol = circ.simulate(trange, v0, i_app)

plt.close("all")

# Plot simulation
plt.figure()
plt.plot(sol.t, sol.y[0], sol.t, sol.y[1], sol.t, sol.y[2])

# Plot I-V curves
V = np.arange(-3,3.1,0.1)
I_passive = V
I_fast = I_passive + i1.out(V)
I_slow = I_fast + i2.out(V) + i3.out(V)
I_ultraslow = I_slow + i4.out(V)
#plt.figure()
#plt.plot(V,I_fast)
#plt.figure()
#plt.plot(V,I_slow)
#plt.figure()
#plt.plot(V,I_ultraslow)
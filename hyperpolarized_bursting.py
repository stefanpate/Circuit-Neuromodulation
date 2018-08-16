# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 16:14:10 2018

Single neuron simulation showcasing hyperpolarization-induced bursting.

@author: Luka
"""

import matplotlib.pyplot as plt

from model import Circuit, LocalizedConductance

# Define current elements
i1 = LocalizedConductance(-2, 0, 'fast') # fast -ve conductance
i2 = LocalizedConductance(2, 0, 'slow') # slow +ve conductance
# slow -ve conductance with inactivation
i3 = LocalizedConductance(-1.6, -0.88, 'slow', -0.5)
i4 = LocalizedConductance(2, 0, 'ultraslow') # ultraslow +ve conductance

# Interconnect the elements
circ = Circuit(i1, i2, i3, i4)

# Simulate the circuit
trange = (0, 20000)
v0 = (-1, -1, -1, -1)

def i_app(t):
    return (t<=10000)*(-2.2) + (t>10000)*(-2.5)

sol = circ.simulate(trange, v0, i_app)

# Plot simulation
plt.close("all")
plt.figure()
plt.plot(sol.t, sol.y[0], sol.t, i_app(sol.t))
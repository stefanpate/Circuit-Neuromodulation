# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 16:14:10 2018

Post-inhibitory rebound in a bursting circuit.

@author: Luka
"""

import matplotlib.pyplot as plt
import numpy as np

from model import LocalizedConductance, Circuit

# Define current elements
i1 = LocalizedConductance(-2, 0, 'fast') # fast-ve conductance
i2 = LocalizedConductance(2, 0, 'slow') # slow +ve conductance
i3 = LocalizedConductance(-1.5, -1.5, 'slow') # slow -ve conductance
i4 = LocalizedConductance(1.5, -1.5, 'ultraslow') # ultraslow +ve conductance

# Interconnect the elements
circ = Circuit(i1, i2, i3, i4)

# Simulate the circuit
Tmax = 20000
trange = (0, Tmax)
v0 = (-2, -2, -2.1, -2.1)

i_base = -2.5
delI = -0.5
i_app = lambda t: (0<=t)*i_base + (t>=Tmax/3)*delI - (t>=2*Tmax/3)*delI
sol = circ.simulate(trange, v0, i_app)

plt.close("all")

# Plot simulation
plt.figure()
plt.plot(sol.t, sol.y[0], sol.t, i_app(sol.t))
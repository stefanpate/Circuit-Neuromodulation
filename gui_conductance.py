# -*- coding: utf-8 -*-
"""
Created on Jul 23 2020

Graphical interface for generating and modulating single neuron behavior.
Neuron consists of 4 conductance elements with single activation variables
representing fast -ve, slow +ve, slow -ve, and ultra-slow +ve conductance.

@author: Luka
"""

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np

from time import time
from collections import deque

from neuron_model import Gate, ConductanceElement, Neuron

plt.ion()

# Initial conductance parameters
g1 = 0
E_rev1 = 30 # 'sodium'
voff1 = -20
k1 = 0.1

g2 = 0
E_rev2 = -75 # 'potassium'
voff2 = -20
k2 = 0.1

g3 = 0
E_rev3 = 140 # 'calcium'
voff3 = -50
k3 = 0.15

g4 = 0
E_rev4 = -75 # 'potassium'
voff4 = -50
k4 = 0.15

# Initial constant input current
i_app_const = -50
i_app = lambda t: i_app_const

# Initial values pulse
pulse_on = False
tend = 0

# Initialize pause value
pause_value = False

# Define timescales
tf = 0
ts = 50
tus = 50*50

# Define activation gates
x1 = Gate(k1, voff1, tf)
x2 = Gate(k2, voff2, ts)
x3 = Gate(k3, voff3, ts)
x4 = Gate(k4, voff4, tus)

# Define conductance elements
i1 = ConductanceElement(g1, E_rev1, x1)
i2 = ConductanceElement(g2, E_rev2, x2)
i3 = ConductanceElement(g3, E_rev3, x3)
i4 = ConductanceElement(g4, E_rev4, x4)

# Interconnect the elements
neuron = Neuron(i1, i2, i3, i4)

def update_fast_vector():
    # Create a list of sections for the fast I-V curve
    
    global fast_vector # List of fast sections
    global fast_index1, fast_index2 # Region of negative conductance
    
    fast_vector = [] 
    
    # Find points where the slope changes
    dIdV = np.diff(I_fast)
    indices = np.where(np.diff(np.sign(dIdV)) != 0)
    indices = indices[0]
    
    indices = np.append(indices, V.size - 1)
    
    # In case the curve is monotone
    fast_index1 = -1
    fast_index2 = -1  

    prev = 0
    slope = dIdV[0] > 0
    
    # Iterate through each section
    for i in np.nditer(indices):
        if slope:
            fast_vector.append([prev, i+2, 'C0']) # Fast +ve
        else:
            fast_vector.append([prev, i+2, 'C3']) # Fast -ve
            fast_index1 = prev
            fast_index2 = i
        slope = not(slope) # Slope changes at each section
        prev = i+1
        

def update_slow_vector():
    # Create a list of sections for the slow/ultra-slow I-V curves
    
    global slow_vector # List of slow sections
    
    slow_vector = [] 
    
    # Find points where slope changes
    dIdV = np.diff(I_slow)
    indices = np.where(np.diff(np.sign(dIdV)) != 0)
    indices = indices[0]
    
    indices = np.append(indices, V.size - 1)
    
    prev = 0
    slope = dIdV[0] > 0
    
    # Iterate through each section    
    for i in np.nditer(indices):
        if slope:
            # Slow +ve, plot regions of fast -ve with different color
            i1 = fast_index1
            i2 = fast_index2
            if i1 < prev:
                i1 = prev-1
            if i1 > i:
                i1 = i-1
            if i2 < prev:
                i2 = prev-1
            if i2 > i:
                i2 = i-1
                
            slow_vector.append([prev, i1+2, 'C0']) # Slow +ve, fast +ve
            slow_vector.append([i1+1, i2+2, 'C3']) # Slow +ve, fast -ve
            slow_vector.append([i2+1, i+2, 'C0']) # Slow +ve, fast +ve            
            
        else:
            slow_vector.append([prev, i+2, 'C1']) # Slow -ve
        slope = not(slope)
        prev = i+1
        
# Plot from the lists of sections       
def plot_fast():
    axf.cla()
    axf.set_xlabel('V')
    axf.set_ylabel('I')
    axf.set_title('Fast')

    for el in fast_vector:
        i1 = el[0]
        i2 = el[1]
        col = el[2]
        axf.plot(V[i1:i2], I_fast[i1:i2], col)
        
def plot_slow():
    axs.cla()
    axs.set_xlabel('V')
    axs.set_ylabel('I')
    axs.set_title('Slow')

    for el in slow_vector:
        i1 = el[0]
        i2 = el[1]
        col = el[2]
        axs.plot(V[i1:i2], I_slow[i1:i2], col)        

def plot_ultra_slow():
    axus.cla()
    axus.set_xlabel('V')
    axus.set_ylabel('I')
    axus.set_title('Ultra-slow')

    for el in slow_vector:
        i1 = el[0]
        i2 = el[1]
        col = el[2]
        axus.plot(V[i1:i2], I_ultraslow[i1:i2], col)
    
    axus.plot(V, np.ones(len(V)) * i_app_const,'C2')

def update_iapp(val):
    global i_app_const, i_app
    i_app_const = val
    i_app = lambda t: i_app_const
    plot_ultra_slow()

def update_fast1(val):
    global i1, I_fast, I_slow, I_ultraslow
    i1.g_max = val
    I_fast = I_passive + i1.ss_out(V)
    I_slow = I_fast + i2.ss_out(V) + i3.ss_out(V)
    I_ultraslow = I_slow + i4.ss_out(V)
    
    update_fast_vector()
    update_slow_vector()
    
    plot_fast()
    plot_slow()
    plot_ultra_slow()
    
def update_fast2(val):
    global i1, I_fast, I_slow, I_ultraslow
    i1.gates[0].voff = val
    I_fast = I_passive + i1.ss_out(V)
    I_slow = I_fast + i2.ss_out(V) + i3.ss_out(V)
    I_ultraslow = I_slow + i4.ss_out(V)

    update_fast_vector()
    update_slow_vector()
    
    plot_fast()
    plot_slow()
    plot_ultra_slow()

def update_slow11(val):
    global i2, I_fast, I_slow, I_ultraslow
    i2.g_max = val
    I_slow = I_fast + i2.ss_out(V) + i3.ss_out(V)
    I_ultraslow = I_slow + i4.ss_out(V)
    
    update_slow_vector()
    
    plot_slow()
    plot_ultra_slow()
    
def update_slow12(val):
    global i2, I_fast, I_slow, I_ultraslow
    i2.gates[0].voff = val
    I_slow = I_fast + i2.ss_out(V) + i3.ss_out(V)
    I_ultraslow = I_slow + i4.ss_out(V)
    
    update_slow_vector()
    
    plot_slow()
    plot_ultra_slow()
    
def update_slow21(val):
    global i3, I_fast, I_slow, I_ultraslow
    i3.g_max = val
    I_slow = I_fast + i2.ss_out(V) + i3.ss_out(V)
    I_ultraslow = I_slow + i4.ss_out(V)
    
    update_slow_vector()
    
    plot_slow()
    plot_ultra_slow()

def update_slow22(val):
    global i3, I_fast, I_slow, I_ultraslow
    i3.gates[0].voff = val
    I_slow = I_fast + i2.ss_out(V) + i3.ss_out(V)
    I_ultraslow = I_slow + i4.ss_out(V)
    
    update_slow_vector()
    
    plot_slow()
    plot_ultra_slow()

def update_ultraslow1(val):
    global i4, I_fast, I_slow, I_ultraslow
    i4.g_max = val
    I_ultraslow = I_slow + i4.ss_out(V)
        
    plot_ultra_slow()

def update_ultraslow2(val):
    global i4, I_fast, I_slow, I_ultraslow
    i4.gates[0].voff = val
    I_ultraslow = I_slow + i4.ss_out(V)

    plot_ultra_slow()
    
# Input pulse event
def pulse(event):
    global pulse_on, tend, i_app
    
    # Pulse parameters
    delta_t = 10
    delta_i = 1
    
    tend = t + delta_t
    pulse_on = True
    
    i_app = lambda t: (i_app_const + delta_i)
    
def pause(event):
    global pause_value
    pause_value = not(pause_value)
    if pause_value:
        button_pause.label.set_text('Resume')
    else:
        button_pause.label.set_text('Pause')

# Plot I-V curves
V = np.arange(-100,20,1.0)
I_passive = V
I_fast = I_passive + i1.ss_out(V)
I_slow = I_fast + i2.ss_out(V) + i3.ss_out(V)
I_ultraslow = I_slow + i4.ss_out(V)

# Find initial sections
update_fast_vector()
update_slow_vector()

# Close pre-existing figures
plt.close("all")

fig = plt.figure()

# Fast I-V curve
axf = fig.add_subplot(2, 3, 1)
axf.set_position([0.1, 0.75, 0.2, 0.2])
plot_fast()

# Slow I-V curve
axs = fig.add_subplot(2, 3, 2)
axs.set_position([0.4, 0.75, 0.2, 0.2])
plot_slow()

# Ultraslow I-V curve
axus = fig.add_subplot(2, 3, 3)
axus.set_position([0.7, 0.75, 0.2, 0.2])
plot_ultra_slow()

# Time - Voltage plot
axsim = fig.add_subplot(2, 3, 4)
axsim.set_position([0.1, 0.45, 0.8, 0.2])
axsim.set_ylim((-120, 20))
axsim.set_xlabel('Time')
axsim.set_ylabel('V')
#axsim.grid()

# Sliders for fast negative conductance
axf1 = plt.axes([0.1, 0.3, 0.3, 0.03])
slider_fast1 = Slider(axf1, '$g_{max}$', 0, 10, valinit = g1)
slider_fast1.on_changed(update_fast1)
axf2 = plt.axes([0.1, 0.25, 0.3, 0.03])
slider_fast2 = Slider(axf2, '$V_{off}$', -75, 0, valinit = voff1)
slider_fast2.on_changed(update_fast2)

# Sliders for slow positive conductance
axs11 = plt.axes([0.1, 0.15, 0.3, 0.03])
slider_slow11 = Slider(axs11, '$g_{max}$', 0, 10, valinit = g2)
slider_slow11.on_changed(update_slow11)
axs12 = plt.axes([0.1, 0.1, 0.3, 0.03])
slider_slow12 = Slider(axs12, '$V_{off}$', -75, 0, valinit = voff2)
slider_slow12.on_changed(update_slow12)

# Sliders for slow negative conductance
axs21 = plt.axes([0.6, 0.3, 0.3, 0.03])
slider_slow21 = Slider(axs21, '$g_{max}$', 0, 10, valinit = g3)
slider_slow21.on_changed(update_slow21)
axs22 = plt.axes([0.6, 0.25, 0.3, 0.03])
slider_slow22 = Slider(axs22, '$V_{off}$', -75, 0, valinit = voff3)
slider_slow22.on_changed(update_slow22)

# Sliders for ultraslow positive conductance
axus1 = plt.axes([0.6, 0.15, 0.3, 0.03])
slider_ultraslow1 = Slider(axus1, '$g_{max}$', 0, 10, valinit = g4)
slider_ultraslow1.on_changed(update_ultraslow1)
axus2 = plt.axes([0.6, 0.1, 0.3, 0.03])
slider_ultraslow2 = Slider(axus2, '$V_{off}$', -75, 0, valinit = voff4)
slider_ultraslow2.on_changed(update_ultraslow2)

# Slider for Iapp
axiapp = plt.axes([0.1, 0.02, 0.5, 0.03])
slider_iapp = Slider(axiapp, '$I_{app}$',-100, 0, valinit = i_app_const)
slider_iapp.on_changed(update_iapp)

# Button for I_app = pulse(t)
axpulse_button = plt.axes([.675, 0.02, 0.1, 0.03])
pulse_button = Button(axpulse_button, 'Pulse')
pulse_button.on_clicked(pulse)

# Button for pausing the simulation
axbutton = plt.axes([0.8, 0.02, 0.1, 0.03])
button_pause = Button(axbutton, 'Pause')
button_pause.on_clicked(pause)

# Labels for conductance sliders
plt.figtext(0.25, 0.34, 'Fast -ve', horizontalalignment = 'center')
plt.figtext(0.25, 0.19, 'Slow +ve', horizontalalignment = 'center')
plt.figtext(0.75, 0.34, 'Slow -ve', horizontalalignment = 'center')
plt.figtext(0.75, 0.19, 'Ultraslow +ve', horizontalalignment = 'center')

# Live simulation
t0 = 0
v0 = neuron.get_init_conditions()

sstep = 100 # draw sstep length of data in a single call
tint = 5000 # time window plotted

tdata, ydata = deque(), deque()
simuln, = axsim.plot(tdata, ydata)

# Define ODE equation for the solvers
def odesys(t, y):
    return neuron.sys(i_app(t),y)

# Standard ODE solvers (RK45, BDF, etc) (import from scipy.integrate)
#solver = BDF(odesys, 0, v0, np.inf, max_step=sstep)
#y = solver.y
#t = solver.t

# Basic Euler step
y = v0
t = t0
def euler_step(odesys, t0, y0):
    dt = 0.1 # step size
    y = y0 + odesys(t0,y0)*dt
    t = t0 + dt
    return t, y
 
# Comment Euler step or standard solver step depending on the method
while plt.fignum_exists(fig.number):
    while pause_value:
        plt.pause(0.01)
    
    st = time()

    last_t = t
    
    # Check for pulse
    while t - last_t < sstep:
        if pulse_on and (t > tend):
            i_app = lambda t: i_app_const
            pulse_on = False
        
        # Euler step
        t,y = euler_step(odesys,t,y)
        
        # Standard solver step
#        msg = solver.step()
#        t = solver.t
#        y = solver.y
#        if msg:
#            raise ValueError('solver terminated with message: %s ' % msg)
        
        tdata.append(t)
        ydata.append(y[0])

    while tdata[-1] - tdata[0] > 2 * tint:
        tdata.popleft()
        ydata.popleft()

    simuln.set_data(tdata, ydata)
    axsim.set_xlim(tdata[-1] - tint, tdata[-1] + tint / 20)
    fig.canvas.draw()
    fig.canvas.flush_events()
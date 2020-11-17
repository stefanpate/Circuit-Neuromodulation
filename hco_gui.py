# -*- coding: utf-8 -*-
"""
Graphical interface for controlling the 2-neuron network

@author: Luka
"""

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, RadioButtons
import numpy as np

from time import time
from collections import deque

from neuron_model import Resistor, CurrentElement, Neuron
from network_model import CurrentSynapse, Network

# Define timescales
tf = 0
ts = 50
tus = 50*50

# List of initial conductance parameters
Rvals = [1, 1]
a_f = [-2, -2]
voff_f = [0, 0]
a_s1 = [2, 2]
voff_s1 = [0, 0]
a_s2 = [-1.5, -1.5]
voff_s2 = [-0.88, -0.88]
a_us = [1.5, 1.5]
voff_us = [-0.88, -0.88]

# Initial constant input current
i_app_const = [-1, -1]

i_app = lambda t: i_app_const

# Initial values pulse
pulse_on = False
tend = 0

# Initialize pause value
pause_value = False

# Define list of circuit elements:
# i1 = fast -ve, i2 = slow +ve, i3 = slow -ve, i4 = ultraslow +ve conductance
R = []
i1 = []
i2 = []
i3 = []
i4 = []

# Define list with all neurons
neurons = []

for j in range(2):
    R.append(Resistor(Rvals[j]))
    i1.append(CurrentElement(a_f[j], voff_f[j], tf))
    i2.append(CurrentElement(a_s1[j], voff_s1[j], ts))
    i3.append(CurrentElement(a_s2[j], voff_s2[j], ts))
    i4.append(CurrentElement(a_us[j], voff_us[j], tus))
    neuron = Neuron(R[j], i1[j], i2[j], i3[j], i4[j])
    neurons.append(neuron)

# Initial synapse parameters
syn1_sign = -1
g1 = [[0, 0], [0, 0]]
syn1 = CurrentSynapse(syn1_sign, -1, ts)

syn2_sign = -1
g2 = [[0, 0], [0, 0]]
syn2 = CurrentSynapse(syn2_sign, -1, ts)

# Define the network
network = Network(neurons, (syn1, g1), (syn2, g2))

def update_fast_vector(j):
    # Create a list of sections for the fast I-V curve for j-th neuron
    
    global fast_vector # List of fast sections
    global fast_index1, fast_index2 # Region of negative conductance
    
    fast_vector[j] = [] 
    
    # Find points where the slope changes
    dIdV = np.diff(I_fast[j])
    indices = np.where(np.diff(np.sign(dIdV)) != 0)
    indices = indices[0]
    
    indices = np.append(indices, V.size - 1)
    
    # In case the curve is monotone
    fast_index1[j] = -1
    fast_index2[j] = -1  

    prev = 0
    slope = dIdV[0] > 0
    
    # Iterate through each section
    for i in np.nditer(indices):
        if slope:
            fast_vector[j].append([prev, i+2, 'C0']) # Fast +ve
        else:
            fast_vector[j].append([prev, i+2, 'C3']) # Fast -ve
            fast_index1[j] = prev
            fast_index2[j] = i
        slope = not(slope) # Slope changes at each section
        prev = i+1
        

def update_slow_vector(j):
    # Create a list of sections for the slow/ultra-slow I-V curves
    
    global slow_vector # List of slow sections
    
    slow_vector[j] = [] 
    
    # Find points where slope changes
    dIdV = np.diff(I_slow[j])
    indices = np.where(np.diff(np.sign(dIdV)) != 0)
    indices = indices[0]
    
    indices = np.append(indices, V.size - 1)
    
    prev = 0
    slope = dIdV[0] > 0
    
    # Iterate through each section    
    for i in np.nditer(indices):
        if slope:
            # Slow +ve, plot regions of fast -ve with different color
            i1 = fast_index1[j]
            i2 = fast_index2[j]
            if i1 < prev:
                i1 = prev-1
            if i1 > i:
                i1 = i-1
            if i2 < prev:
                i2 = prev-1
            if i2 > i:
                i2 = i-1
                
            slow_vector[j].append([prev, i1+2, 'C0']) # Slow +ve, fast +ve
            slow_vector[j].append([i1+1, i2+2, 'C3']) # Slow +ve, fast -ve
            slow_vector[j].append([i2+1, i+2, 'C0']) # Slow +ve, fast +ve            
            
        else:
            slow_vector[j].append([prev, i+2, 'C1']) # Slow -ve
        slope = not(slope)
        prev = i+1
        
# Plot from the lists of sections       
def plot_fast(j):
    axf[j].cla()
    axf[j].set_xlabel('V')
    axf[j].set_ylabel('I')
    axf[j].set_title('Fast')

    for el in fast_vector[j]:
        i1 = el[0]
        i2 = el[1]
        col = el[2]
        axf[j].plot(V[i1:i2], I_fast[j][i1:i2], col)
        
def plot_slow(j):
    axs[j].cla()
    axs[j].set_xlabel('V')
    axs[j].set_ylabel('I')
    axs[j].set_title('Slow')

    for el in slow_vector[j]:
        i1 = el[0]
        i2 = el[1]
        col = el[2]
        axs[j].plot(V[i1:i2], I_slow[j][i1:i2], col)        

def plot_ultra_slow(j):
    axus[j].cla()
    axus[j].set_xlabel('V')
    axus[j].set_ylabel('I')
    axus[j].set_title('Ultra-slow')

    for el in slow_vector[j]:
        i1 = el[0]
        i2 = el[1]
        col = el[2]
        axus[j].plot(V[i1:i2], I_ultraslow[j][i1:i2], col)
    
    axus[j].plot(V, np.ones(len(V)) * i_app_const[j],'C2')

def update_iapp(val, j):
    global i_app_const
    i_app_const[j] = val
    plot_ultra_slow(j)

def update_fast1(val, j):
    global i1, I_fast, I_slow, I_ultraslow
    i1[j].a = -val
    I_fast[j] = I_passive[j] + i1[j].out(V)
    I_slow[j] = I_fast[j] + i2[j].out(V) + i3[j].out(V)
    I_ultraslow[j] = I_slow[j] + i4[j].out(V)
    
    update_fast_vector(j)
    update_slow_vector(j)
    
    plot_fast(j)
    plot_slow(j)
    plot_ultra_slow(j)
    
def update_fast2(val, j):
    global i1, I_fast, I_slow, I_ultraslow
    i1[j].voff = val
    I_fast[j] = I_passive[j] + i1[j].out(V)
    I_slow[j] = I_fast[j] + i2[j].out(V) + i3[j].out(V)
    I_ultraslow[j] = I_slow[j] + i4[j].out(V)

    update_fast_vector(j)
    update_slow_vector(j)
    
    plot_fast(j)
    plot_slow(j)
    plot_ultra_slow(j)

def update_slow11(val, j):
    global i2, I_fast, I_slow, I_ultraslow
    i2[j].a = val
    I_slow[j] = I_fast[j] + i2[j].out(V) + i3[j].out(V)
    I_ultraslow[j] = I_slow[j] + i4[j].out(V)
    
    update_slow_vector(j)
    
    plot_slow(j)
    plot_ultra_slow(j)
    
def update_slow12(val, j):
    global i2, I_fast, I_slow, I_ultraslow
    i2[j].voff = val
    I_slow[j] = I_fast[j] + i2[j].out(V) + i3[j].out(V)
    I_ultraslow[j] = I_slow[j] + i4[j].out(V)
    
    update_slow_vector(j)
    
    plot_slow(j)
    plot_ultra_slow(j)
    
def update_slow21(val, j):
    global i3, I_fast, I_slow, I_ultraslow
    i3[j].a = -val
    I_slow[j] = I_fast[j] + i2[j].out(V) + i3[j].out(V)
    I_ultraslow[j] = I_slow[j] + i4[j].out(V)
    
    update_slow_vector(j)
    
    plot_slow(j)
    plot_ultra_slow(j)

def update_slow22(val, j):
    global i3, I_fast, I_slow, I_ultraslow
    i3[j].voff = val
    I_slow[j] = I_fast[j] + i2[j].out(V) + i3[j].out(V)
    I_ultraslow[j] = I_slow[j] + i4[j].out(V)
    
    update_slow_vector(j)
    
    plot_slow(j)
    plot_ultra_slow(j)

def update_ultraslow1(val, j):
    global i4, I_fast, I_slow, I_ultraslow
    i4[j].a = val
    I_ultraslow[j] = I_slow[j] + i4[j].out(V)
        
    plot_ultra_slow(j)

def update_ultraslow2(val, j):
    global i4, I_fast, I_slow, I_ultraslow
    i4[j].voff = val
    I_ultraslow[j] = I_slow[j] + i4[j].out(V)

    plot_ultra_slow(j)
    
def update_syn1(val):
    if val == 'Inhibitory':
        syn1.sign = -1
    if val == 'Excitatory':
        syn1.sign = +1

def update_syn2(val):
    if val == 'Inhibitory':
        syn2.sign = -1
    if val == 'Excitatory':
        syn2.sign = +1
        
def update_syn1_gain(val):
    # g1 = network.synapses[0][1], change g1[0][1] to val
    network.synapses[0][1][0][1] = val

def update_syn2_gain(val):
    # g2 = network.synapses[1][1], change g2[1][0] to val
    network.synapses[1][1][1][0] = val
        
update_iappl = [(lambda val: update_iapp(val,0)), (lambda val: update_iapp(val,1))]
update_fast1l = [(lambda val: update_fast1(val,0)), (lambda val: update_fast1(val,1))]
update_fast2l = [(lambda val: update_fast2(val,0)), (lambda val: update_fast2(val,1))]
update_slow11l = [lambda val: update_slow11(val,0), lambda val: update_slow11(val,1)]
update_slow12l = [lambda val: update_slow12(val,0), lambda val: update_slow12(val,1)]
update_slow21l = [lambda val: update_slow21(val,0), lambda val: update_slow21(val,1)]
update_slow22l = [lambda val: update_slow22(val,0), lambda val: update_slow22(val,1)]
update_ultraslow1l = [lambda val: update_ultraslow1(val,0), lambda val: update_ultraslow1(val,1)]
update_ultraslow2l = [lambda val: update_ultraslow2(val,0), lambda val: update_ultraslow2(val,1)]

# Input pulse event
def pulse(event):
    global pulse_on, tend, i_app
    
    # Pulse parameters
    delta_t = 500
    delta_i = 1
    
    tend = t + delta_t
    pulse_on = True
    
    i_app = lambda t: [i_app_const[0] - delta_i, i_app_const[1]]
    
def pause(event):
    global pause_value
    pause_value = not(pause_value)
    if pause_value:
        button_pause.label.set_text('Resume')
    else:
        button_pause.label.set_text('Pause')
        
def save(event):
    global out_no
    f = open("data" + str(out_no) + ".txt", "w")
    np.savetxt(f, np.array([tdata, ydata, ydata1]).T)
    plt.savefig("fig" + str(out_no) + ".pdf")
    out_no = out_no + 1
    f.close()
    
# Initialize I-V data
I_passive = []
I_fast = []
I_slow = []
I_ultraslow = []

# Plot I-V curves
V = np.arange(-3,3.1,0.1)

for j in range(2):
    I_passive.append(V)
    I_fast.append(I_passive[j] + i1[j].out(V))
    I_slow.append(I_fast[j] + i2[j].out(V) + i3[j].out(V))
    I_ultraslow.append(I_slow[j] + i4[j].out(V))

# Initialize section vectors
fast_vector = [[], []]
fast_index1 = [[], []]
fast_index2 = [[], []]
slow_vector = [[], []]

# Find initial sections
for j in range(2):
    update_fast_vector(j)
    update_slow_vector(j)

# Close pre-existing figures
plt.close("all")

#fig = plt.figure()
fig = plt.figure(figsize = (10,5)) # default size is (6.4,4.76)

# Initialize axis
axf = [[],[]]
axs = [[],[]]
axus = [[],[]]

for j in range(2):
    # Fast I-V curve
    axf[j] = fig.add_subplot(1, 6, j*3 + 1)
    axf[j].set_position([j*0.5 + 0.05, 0.75, 0.1, 0.17])
    plot_fast(j)

    # Slow I-V curve
    axs[j] = fig.add_subplot(1, 6, j*3 + 2)
    axs[j].set_position([j*0.5 + 0.2, 0.75, 0.1, 0.17])
    plot_slow(j)

    # Ultraslow I-V curve
    axus[j] = fig.add_subplot(1, 6, j*3 + 3)
    axus[j].set_position([j*0.5 + 0.35, 0.75, 0.1, 0.17])
    plot_ultra_slow(j)

## Time - Voltage plot
axsim = fig.add_subplot(1, 1, 1)
axsim.set_position([0.05, 0.42, 0.4, 0.2])
axsim.set_ylim((-5, 5))
axsim.set_xlabel('Time')
axsim.set_ylabel('V')

# Initialize sliders & buttons
axf1 = [[],[]]
slider_fast1 = [[],[]]
axf2 = [[],[]]
slider_fast2 = [[],[]]
axs11 = [[],[]]
slider_slow11 = [[],[]]
axs12 = [[],[]]
slider_slow12 = [[],[]]
axs21 = [[],[]]
slider_slow21 = [[],[]]
axs22 = [[],[]]
slider_slow22 = [[],[]]
axus1 = [[],[]]
slider_ultraslow1 = [[],[]]
axus2 = [[],[]]
slider_ultraslow2 = [[],[]]
axiapp = [[],[]]
slider_iapp = [[],[]]

for j in range(2):
    # Sliders for fast negative conductance
    axf1[j] = plt.axes([j*0.5+0.05, 0.3, 0.15, 0.03])
    slider_fast1[j] = Slider(axf1[j], 'Gain', 0, 6, valinit = -a_f[j])
    slider_fast1[j].on_changed(update_fast1l[j])
    axf2[j] = plt.axes([j*0.5+0.05, 0.25, 0.15, 0.03])
    slider_fast2[j] = Slider(axf2[j], '$V_{off}$', -2, 2, valinit = voff_f[j])
    slider_fast2[j].on_changed(update_fast2l[j])

    # Sliders for slow positive conductance
    axs11[j] = plt.axes([j*0.5+0.05, 0.15, 0.15, 0.03])
    slider_slow11[j] = Slider(axs11[j], 'Gain', 0, 6, valinit = a_s1[j])
    slider_slow11[j].on_changed(update_slow11l[j])
    axs12[j] = plt.axes([j*0.5+0.05, 0.1, 0.15, 0.03])
    slider_slow12[j] = Slider(axs12[j], '$V_{off}$', -2, 2, valinit = voff_s1[j])
    slider_slow12[j].on_changed(update_slow12l[j])
    
    # Sliders for slow negative conductance
    axs21[j] = plt.axes([j*0.5+0.3, 0.3, 0.15, 0.03])
    slider_slow21[j] = Slider(axs21[j], 'Gain', 0, 6, valinit = -a_s2[j])
    slider_slow21[j].on_changed(update_slow21l[j])
    axs22[j] = plt.axes([j*0.5+0.3, 0.25, 0.15, 0.03])
    slider_slow22[j] = Slider(axs22[j], '$V_{off}$', -2, 2, valinit = voff_s2[j])
    slider_slow22[j].on_changed(update_slow22l[j])
    
    # Sliders for ultraslow positive conductance
    axus1[j] = plt.axes([j*0.5+0.3, 0.15, 0.15, 0.03])
    slider_ultraslow1[j] = Slider(axus1[j], 'Gain', 0, 6, valinit = a_us[j])
    slider_ultraslow1[j].on_changed(update_ultraslow1l[j])
    axus2[j] = plt.axes([j*0.5+0.3, 0.1, 0.15, 0.03])
    slider_ultraslow2[j] = Slider(axus2[j], '$V_{off}$', -2, 2, valinit = voff_us[j])
    slider_ultraslow2[j].on_changed(update_ultraslow2l[j])
    
    # Slider for Iapp
    axiapp[j] = plt.axes([0.5*j+0.05, 0.02, 0.3, 0.03])
    slider_iapp[j] = Slider(axiapp[j], '$I_{app}$',-3, 3, valinit = i_app_const[j])
    slider_iapp[j].on_changed(update_iappl[j])
    
    # Labels for conductance sliders
    plt.figtext(j*0.5+0.125, 0.34, 'Fast -ve', horizontalalignment = 'center')
    plt.figtext(j*0.5+0.125, 0.19, 'Slow +ve', horizontalalignment = 'center')
    plt.figtext(j*0.5+0.375, 0.34, 'Slow -ve', horizontalalignment = 'center')
    plt.figtext(j*0.5+0.375, 0.19, 'Ultraslow +ve', horizontalalignment = 'center')
    
# Radio buttons for synapses
axsyn1 = plt.axes([0.55, 0.42, 0.1, 0.2])
radio_button1 = RadioButtons(axsyn1,['Inhibitory', 'Excitatory'], active = 0)
radio_button1.on_clicked(update_syn1)
plt.figtext(0.6, 0.63, 'Synapse 1->2', horizontalalignment = 'center')

axsyn2 = plt.axes([0.66, 0.42, 0.1, 0.2])
radio_button2 = RadioButtons(axsyn2,['Inhibitory', 'Excitatory'], active = 0)
radio_button2.on_clicked(update_syn2)
plt.figtext(0.71, 0.63, 'Synapse 2->1', horizontalalignment = 'center')

# Sliders for synapses
axsynsl1 = plt.axes([0.8, 0.545, 0.15, 0.03])
slider_syn1 = Slider(axsynsl1, 'Gain', 0, 1.5, valinit = g1[0][1])
plt.figtext(0.875, 0.585, 'Synapse 1->2', horizontalalignment = 'center')
slider_syn1.on_changed(update_syn1_gain)

axsynsl2 = plt.axes([0.8, 0.445, 0.15, 0.03])
slider_syn2 = Slider(axsynsl2, 'Gain', 0, 1.5, valinit = g2[1][0])
plt.figtext(0.875, 0.485, 'Synapse 2->1', horizontalalignment = 'center')
slider_syn2.on_changed(update_syn2_gain)

# Button pausing the simulation
axpause_button = plt.axes([.9, .02, 0.08, .06])
button_pause = Button(axpause_button, 'Pause')
button_pause.on_clicked(pause)

# Button for Iapp pulse to Neuron 1
axpulse_button = plt.axes([.4, 0.02, 0.08, 0.06])
pulse_button = Button(axpulse_button, 'Pulse')
pulse_button.on_clicked(pulse)

# Button for saving the data
out_no = 0 # number to attach to the output files
axsave_button = plt.axes([.46, .5, 0.08, .06])
save_button = Button(axsave_button, 'Save')
save_button.on_clicked(save)

# Live simulation
t0 = 0
v0 = network.get_init_conditions()
#v0 = (-1.5, -1.5, -1.4, -1.1, -1.1, -1.2)

sstep = 100
tint = 5000

tdata, ydata, ydata1 = deque(), deque(), deque()
simuln1, simuln2 = axsim.plot(tdata, ydata, tdata, ydata1)

def odesys(t, y):
    return network.sys(i_app(t),y)

# Standard ODE solvers (RK45, BDF, etc) (import from scipy.integrate)
#solver = BDF(odesys, 0, v0, np.inf, max_step=sstep)
#y = solver.y
#t = solver.t

# Basic Euler step
y = v0
t = t0
def euler_step(odesys, t0, y0):
    dt = 1 # step size
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
        ydata1.append(y[3])

    while tdata[-1] - tdata[0] > 2 * tint:
        tdata.popleft()
        ydata.popleft()
        ydata1.popleft()

    simuln1.set_data(tdata, ydata)
    simuln2.set_data(tdata, ydata1)
    axsim.set_xlim(tdata[-1] - tint, tdata[-1] + tint / 20)
    fig.canvas.draw()
    fig.canvas.flush_events()

# -*- coding: utf-8 -*-
"""
Created on Jul 23 2020

Single neuron circuit model. Circuit consists of a parallel interconnection of
an arbitrary number of either 'Current' or 'Conductance' elements

@author: Luka
"""
from numpy import tanh, exp
import numpy as np

# Default initial conditions
vmem0 = -1.9 # Membrane voltage
vx0 = -1.8 # First-order filters

def sigmoid(x, k = 1):
    return 1 / (1 + exp(-k * (x)))

class SingleTimescaleElement():
    """
    Parent class for elements depending on a single filtered voltage Vx
    """
    
    timescale = None
    v0 = None
    v_index = None # index of corresponding Vx, assigned at circuit init
    
    def add_timescale(self, timescales, y0):
        """
        Add a first-order filter for Vx, or associate the element with an
        existing Vx
        """
        
        if (self.timescale in timescales) and (self.v0 is None):
            self.v_index = timescales.index(self.timescale)
        else:
            timescales.append(self.timescale)
            self.v_index = len(timescales) - 1
            if (self.v0):
                y0.append(self.v0)
            else:
                y0.append(vx0)        

class CurrentElement(SingleTimescaleElement):
    """
    Current element of the form:
        Iout = a * tanh(Vx - voff).
        taux * dVx/dt = Vmem - Vx
    """                                
    
    def __init__(self, a, voff, timescale, v0 = None):
        if (timescale == 0) and (v0 is not None):
            raise ValueError("Initial condition of an instantaneous element "
                             "cannot be set")
            
        self.a = a
        self.voff = voff
        self.timescale = timescale
        self.v0 = v0
                
    def out(self, y):        
        return (self.a * tanh(y[self.v_index] - self.voff))
    
    def ss_out(self, V):
        """
        Generate the steady-state input-output characteristic
        """
        return (self.a * tanh(V - self.voff))
    
class Gate(SingleTimescaleElement):
    """
    Single gating variable with sigmoidal activation/inactivation:
        x = S(k*(Vx - voff))
        taux * dVx/dt = Vmem - Vx
    """
    
    def __init__(self, k, voff, timescale, v0 = None):
        if (timescale == 0) and (v0 is not None):
            raise ValueError("Initial condition of an instantaneous element "
                             "cannot be set")
            
        self.k = k
        self.voff = voff
        self.timescale = timescale
        self.v0 = v0
        
    def out(self, y):
        return sigmoid(y[self.v_index] - self.voff, self.k)
    
    def ss_out(self, V):
        """
        Generate the steady-state input-output characteristic
        """
        return sigmoid(V - self.voff, self.k)

class ConductanceElement:
    """
    Single conductance element consisting of multiple gates:
        Iout = g_max * x1 * x2 * ... * xn * (V - E_rev)
        *args: [x1,x2,...,xn] = gates
    """
    def __init__(self, g_max, E_rev, *args):
        self.g_max = g_max
        self.E_rev = E_rev
        self.gates = args
    
    def add_timescale(self, timescales, y0):
        """
        Add timescales for each gating variable
        """
        for x in self.gates:
            x.add_timescale(timescales, y0)
    
    def out(self, y):
        iout = self.g_max * (y[0] - self.E_rev)
        for x in self.gates:
            iout *= x.out(y)
        return iout
    
    def ss_out(self, V):
        """
        Generate the steady-state input-output characteristic
        """
        iout = self.g_max * (V - self.E_rev)
        for x in self.gates:
            iout *= x.ss_out(V)
        return iout        
        
class Neuron:
    """
    Parallel interconnection of current or conductance elements
    C dV/dT = -V/R - sum(I_x) + Iapp
    where I_x is the output current of each current/conductance element
    
    args: list of feedback elements (ion currents)
    kwargs: circuit parameters (membrane R,C values)
    """

    _stdPar = {'R': 1,'C': 1} # Membrane RC values
    timescales = [0] # Timescales of membrane voltage + first order filters
    y0 = [vmem0] # Initial conditions of membrane voltage + first order filters
    
    @property
    def stdPar(self):
        return type(self)._stdPar
    
    def __init__(self, *args, **kwargs):
        self.__dict__.update(self.stdPar) # Default circuit parameters
        self.__dict__.update(kwargs) # Modify circuit parameters
        
        self.elements = args # List containing all feedback elements
        
        # Group instances of same timescale + set initial conditions
        for el in self.elements:
            el.add_timescale(self.timescales, self.y0)
            
    def get_init_conditions(self):
        return np.array(self.y0)
    
    def i_sum(self, y):
        """
        Returns total internal current
        """
        s = y[0] / self.R
        for el in self.elements:
            s += el.out(y)
        return s
    
    def sys(self, i_app, y):
        dy = []
        dvmem = (i_app - self.i_sum(y)) / self.C
        dy.append(dvmem)
        
        # First-order filters
        for index, tau in enumerate(self.timescales[1:]):
            dy.append((y[0] - y[index+1]) / tau)
        
        return np.array(dy)
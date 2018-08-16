# -*- coding: utf-8 -*-
"""
Created on Fri May 25 14:28:30 2018

Single neuron circuit model.
Circuit class consists of a parallel interconnection of an arbitrary number
of LocalizedConductance elements, each representing fast,slow or ultraslow
positive/negative conductance.

@author: Luka
"""
from autograd import jacobian
from autograd.numpy import tanh, exp
import autograd.numpy as np
from scipy.integrate import solve_ivp

def sigmoid(x, k = 1):
    return 1 / (1 + exp(-k * (x)))

class LocalizedConductance:
    """
    Localized conductance element of the form a * tanh(V - voff).
    If the element has inactivation dynamics, v_inact_off needs to be
    specified, and the inactivation variable passed to the function out.
    """
    
    _timescales = ('fast', 'slow', 'ultraslow')
    
    @property
    def timescale(self):
        return self._timescale
    
    @timescale.setter
    def timescale(self, tau):
        if tau in self._timescales:
            self._timescale = tau
        else:
            raise ValueError("Invalid timescale")
                                
    def __init__(self, a, voff, timescale, v_inact_off = None):
        self.a = a
        self.voff = voff
        self.timescale = timescale
        self.v_inact_off = v_inact_off
        
    def out(self, V, V_inact = None):
        if V_inact is not None:
            # Slope of sigmoidal inactivation, determined by implementation
            k = -2
            
            h = sigmoid(V_inact - self.v_inact_off, k)
        else:
            h = 1
        
        return (h * self.a * tanh(V - self.voff))
                
class Circuit:
    """
    Parallel interconnection of localized conductance elements
    CdV/dT = -V - sum(I_x) + Iapp
    """

    _stdPar = {'C': 1, 'tf': 0, 'ts': 50, 'tus': 50**2}
    
    @property
    def stdPar(self):
        return type(self)._stdPar
    
    def __init__(self, *args, **kwargs):
        self.__dict__.update(self.stdPar) # Default circuit parameters
        self.__dict__.update(kwargs) # Modify circuit parameters
        
        self.conductances = args # List containing all localized coductances
    
    def i_sum(self, v, vf, vs, vus):
        
        s = v
        for i in self.conductances:
            if i.timescale == 'fast':
                    s += i.out(vf)
            if i.timescale == 'slow':
                # Check if the slow current has inactivation
                if i.v_inact_off is None:
                    s += i.out(vs)
                else:
                    s += i.out(vs, vus)
            if i.timescale == 'ultraslow':
                s += i.out(vus)                
        return s
    
    def sys(self, *y):
        
        v = y[0]
        i_app = y[-1]
        
        if self.tf != 0:
            vf = y[1]
            vs = y[2]
            vus = y[3]
        else:
            vf = v
            vs = y[1]
            vus = y[2]
        
        dv = (i_app - self.i_sum(v, vf, vs, vus)) / self.C
        dvs = (v - vs) / self.ts
        dvus = (v - vus) / self.tus
        
        if self.tf != 0:
            dvf = (v - vf) / self.tf
            return np.array([dv, dvf, dvs, dvus])
        else:
            return np.array([dv, dvs, dvus])
    
    def simulate(self, trange, v0, i_app):
        
        if self.tf == 0:
            v0 = (v0[0], v0[2], v0[3])
        
        def odesys(t, y):
            return self.sys(*y, i_app(t))
        def odejac(t, y):
            return jacobian(lambda y: odesys(t, y))(y)
        
        sol = solve_ivp(odesys, trange, v0, method='BDF', dense_output=True,
                        jac=odejac)
        return sol
    
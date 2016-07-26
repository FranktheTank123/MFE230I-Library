'''
This library contains Monte Carlo-related functions

Modification log:


7/22/2016 - Initiation of the script
'''

import numpy as np
import scipy as sp
from scipy.stats import norm
import Fixed_Income_Library as fi


def multiStepMC(z, price_evolution, anti = False
                , tracker = lambda S_ts : S_ts):
    '''
    multi-step-mc:
    ***NOTE THE STEPS IS DETERMINED BY THE DIMENSION OF Z (which is a np.array)***

    assume equally spaced time steps


    price_evolution: a function that takes an 1d array of Z slice and
                     returns 1d array (+1 size to include s0) of the evlotion
                     of underlyings which based on the Zs

    tracker: a function (takes an array of evolution of underlyings)
            that keep track of features of the evolution of the
            stock price, which could be max/min, or whether a boundary is hitted
    '''

    if anti:
        z = -z

    ## generate the evolution of underlyings for all pathes
    evolutions_ = np.apply_along_axis(price_evolution, 1, z)

    return evolutions_[:,-1], np.apply_along_axis(tracker, 1, evolutions_)

def hullWhiteMC(theta, kappa, sig, rt, T, z, t = 0):
    '''
    Monte Carlo simulation on Hull White model, from t to T

    parameters --
    theta: function of t
    kappa, sig:  constant
    rt: initial starting point
    T:  terminal time
    z: 1d array of normal random variable
    t: staring time
    '''

    dt_ = (T-t)/len(z)

    time_stpes_ = np.linspace(t,T,len(z)+1) ## time space
    r_s_ = np.repeat(rt, len(z)+1)          ## r_t space

    for i, z_i in enumerate(z):
        r_s_[i+1] = r_s_[i]+(theta(time_stpes_[i+1])-kappa*r_s_[i])\
                    * dt_ + sig*dt_**0.5 * z_i
    #print(dt_,time_stpes_)
    return r_s_

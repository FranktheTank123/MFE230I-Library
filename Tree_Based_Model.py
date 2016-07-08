'''
This library contains tree-based-model-related functions


Modification log:


7/7/2016 - Initiation of the script
'''


import pandas as pd
import numpy as np
import scipy as sp
from scipy.optimize import minimize
import Fixed_Income_Library as fi
import Duration_Convexity as dc


class HoLeeTree:
    '''
    the class will construct a simple Ho-Lee tree
    '''
    def __init__(self, sig, T, dt = 1, q = 0.5):
        '''
        Initiation parameters -
        sig:    the constant volatility should be pre-defined
        T:      total level of trees
        dt:     each time step
        q:      the risk-neutual probability, assume to be 0.5 by default
        '''
        self.dt = dt
        self.q = q
        self.sig = sig
        self.T = T


    def getDriftPrice( self, drifts ):
        '''
        drifts: a Tx1 vector indicate the drift at each level of the tree

        will return a Tx1 price ( assume par = 1) vector, using the given drifts
        '''

        T = self.T
        if (len(drifts) != T):
            raise Exception("Tree size={}, while drift size={}".format(T, len(drifts)))

        sig = self.sig
        dt = self.dt
        q = self.q

        prices_pred_ = np.zeros(T)

        ind_ = 0
        for time_ in range(1, T + 1): # time_ = 1, 2, 3, ..., T
             # the last step
            temp_price_ = np.repeat( 1, time_) / (1 + drifts[:time_].sum() + \
                    np.array([ (time_-1-2*x) * sig for x in range(time_)])) ** dt

            # trace back the tree until it hits the first period
            while( time_ > 1):
                time_ -= 1
                temp_price_ = (q*temp_price_[:-1] + (1-q)*temp_price_[1:]) / \
                    (1 + drifts[:time_].sum() + \
                    np.array([ (time_-1-2*x) * sig for x in range(time_)])) ** dt

            # append the results
            prices_pred_[ ind_] = temp_price_[0]
            ind_ += 1

        return prices_pred_


    def calibrate_drift(self, prices, drift_init, method = 'nelder-mead', options = {'xtol': 1e-8, 'disp': False} ):
        '''
        prices: normalized to per dollar
        drift_init: initial value to optimize
        method: the optimization method of the cost function
        options: used for minimize function
        '''
        if (len(prices) != self.T):
            raise Exception("Tree size={}, while prices size={}".format(T, len(prices)))


        getDriftPrice_wrapper = lambda x: ((self.getDriftPrice(x) - prices) ** 2).sum()

        results_  = minimize( getDriftPrice_wrapper, drift_init
                            , method = method, options = options)
        self.drifts = results_.x





#

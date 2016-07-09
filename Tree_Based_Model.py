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
    this could also be (possibly) generalized into BDT tree later..
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
        self.drifts = np.repeat(0.01, T) # just a place holder here...

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


        # personally, I don't like loop, but I cound't find a better way here...
        for (ind_, time_) in  enumerate(range(1, T + 1)): # time_ = 1, 2, 3, ..., T
             # the last step, temp_price_ will be a 1xn vector
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

        return prices_pred_


    def calibrateDrift(self, prices, drift_init, method = 'nelder-mead'
                        , options = {'xtol': 1e-8, 'disp': False} ):
        '''
        Calibrate the dirft vector using the market ZCB prices (assume no coupon payment here)

        Parameters --
        prices:         normalized to per dollar
        drift_init:     initial value to optimize
        method:         the optimization method of the cost function
        options:        used for minimize function
        '''
        if (len(prices) != self.T):
            raise Exception("Tree size={}, while prices size={}".format(T, len(prices)))


        getDriftPrice_wrapper = lambda x: ((self.getDriftPrice(x) - prices) ** 2).sum()

        results_  = minimize( getDriftPrice_wrapper, drift_init
                            , method = method, options = options)
        self.drifts = results_.x


    def plotIRTree(self):
        '''
        generate a TxT upper-triangle matrix, representing the tree of the IR
        '''
        tree_ = np.zeros(self.T**2).reshape(self.T,self.T)
        for t_ in range(self.T):
            tree_[:t_ + 1, t_] = self.drifts[:t_ + 1].sum() + \
                        np.array([ (t_-2*x) * self.sig for x in range(t_ + 1)])
        self.IRtree = tree_



def irTreeToPayoffTree( Tree, get_cash_flow, get_curr_price = lambda T, p: p):
    '''
    This is a very generic function, that return a payoff tree matrix with
    size = IRtree

    Parameters --
    Tree:           an interest rate Tree object
    get_cash_flow:  a function that return the cashflow at a specifc time T
    get_curr_price: a function that take two parameters (T, curr_price),
                    and decide the actual current price (incorprate early exercise).
                    By default, it's European
    '''
    ## chekc if Tree object is valid
    try:
        IRTree = Tree.IRtree
        dt = Tree.dt
        q = Tree.q
        T = Tree.T
    except:
        raise Exception('The Tree object provided is not valid.')

    ## check tree size
    if (IRTree.shape != (T, T)):
        raise Exception('Invalid IRTree size={}'.formate(IRTree.shape))

    payoff_tree_ = np.zeros(T**2).reshape(T, T) ## Initiation

    ## The last Column case is a bit special:
    europrice_ = get_cash_flow(T*dt) / (1+IRTree[:,T-1])**dt
    payoff_tree_[:,T-1] = get_curr_price(T*dt-dt, europrice_)

    for t_ in range(T-2, -1, -1): ## t_ = T-2, T-1, .., 0
        europrice_ = (get_cash_flow(t_*dt) + q * payoff_tree_[:t_+1, t_+1] \
                + (1-q) * payoff_tree_[1:t_+2, t_+1] )\
                / (1+IRTree[:t_+1, t_])**dt
        payoff_tree_[:t_+1,t_] = get_curr_price(t_*dt-dt, europrice_)

    return payoff_tree_

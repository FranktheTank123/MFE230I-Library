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


class hoLeeTree:
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
             # the last step, temp_price_ will be a 1xn+1 vector
            temp_price_ = np.repeat( 1, time_+1)

            # trace back the tree until it hits the first period
            while( time_ > 0):
                temp_price_ = (q*temp_price_[:-1] + (1-q)*temp_price_[1:]) / \
                    (1 + drifts[:time_].sum() + \
                    np.array([ (time_-1-2*x) * sig for x in range(time_)]))
                time_ -= 1
            # append the results
            prices_pred_[ ind_] = temp_price_[0]

        return prices_pred_


    def calibrateDrift(self, prices, drift_init, method = 'nelder-mead'
                        , options = {'xtol': 1e-8, 'disp': True} ):
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
        tree_ = np.repeat(np.nan,self.T**2).reshape(self.T,self.T)
        for t_ in range(self.T):
            tree_[:t_ + 1, t_] = self.drifts[:t_ + 1].sum() + \
                        np.array([ (t_-2*x) * self.sig for x in range(t_ + 1)])
        self.IRtree = tree_


class bdtTree:
    '''
    the class will construct a BDT tree
    '''
    def __init__(self, T, dt = 1, q = 0.5):
        '''
        Initiation parameters -
        T:      total level of trees
        dt:     each time step
        q:      the risk-neutual probability, assume to be 0.5 by default
        '''
        self.dt = dt
        self.q = q
        self.T = T
        #self.para =  np.repeat(0.01, 2*T/dt-1)  # just a place holder here...
        self.drifts = np.repeat(0.01, T/dt)
        self.sigs = np.repeat(0, T/dt)

    def getDriftPrice( self, drifts ):
        '''
        para: a (T+T-1)x1 vector indicate the drift&sig at each level of the tree
        will return a Tx1 price ( assume par = 1) vector, using the given drifts
        '''

        T = self.T
        dt = self.dt
        if (len(drifts) != T/dt):
            raise Exception("drifts size={},should be {}".format(len(drifts),T/dt))

        #drifts = para[0:T/dt]
        sigs = self.sigs
        q = self.q

        prices_pred_ = np.zeros(T/dt)

        #print("drifts size =", len(drifts))
        #print("sigs size=",len(sigs))

        for (ind_, time_) in  enumerate(range(1, int(T/dt) + 1)): # time_ = 1, 2, 3, ..., T
            #print(ind_, time_)
            ## the last step, temp_price_ will be a 1xn+1 vector
            temp_price_ = np.repeat( 1, time_+1)
            ## trace back the tree until it hits the first period
            while( time_ > 0):
                temp_price_ = (q*temp_price_[:-1] + (1-q)*temp_price_[1:]) / ( 1 + dt * \
                         np.exp(drifts[:time_].sum()) * np.exp([ 2* x * sigs[time_-1] * (dt**0.5) for x in range(time_ -1 ,-1,-1)]))
                time_ -= 1
            ## append the results
            prices_pred_[ ind_] = temp_price_[0]

        return prices_pred_


    def calibrateDrift(self, prices, sigs, drift_init, method = 'nelder-mead'
                        , options = {'xtol': 1e-8, 'disp': True} ):
        '''
        Calibrate the dirft vector using the market ZCB prices (assume no coupon payment here)

        Parameters --
        prices:         normalized to per dollar
        sigs:           volatility of the prices
        drift_init:     initial value to optimize
        method:         the optimization method of the cost function
        options:        used for minimize function
        '''
        if (len(prices) != self.T/self.dt):
            raise Exception("Tree size={}, while prices size={}".format(self.T/self.dt, len(prices)))

        if( len(drift_init) != len(self.drifts)):
            raise Exception("input drift size={}, should be {}".format(len(drift_init),len(self.drifts)))

        self.sigs = np.append(0,sigs[:-1])

        getDriftPrice_wrapper = lambda x: ((self.getDriftPrice(x) - prices) ** 2).sum()

        results_  = minimize( getDriftPrice_wrapper, drift_init
                            , method = method, options = options)
        #self.para = results_.x
        self.drifts = results_.x
        #self.sigs = np.append(0,self.para[self.T/self.dt:])


    def plotIRTree(self):
        '''
        generate a T/dt x T/dt upper-triangle matrix, representing the tree of the IR
        '''

        tree_ = np.repeat(np.nan,(self.T/self.dt)**2).reshape((self.T/self.dt),(self.T/self.dt))
        for t_ in range(1, int(self.T/self.dt) + 1):
            tree_[:t_, t_-1] = np.exp(self.drifts[:t_].sum()) * \
                np.exp([ 2* x * self.dt**0.5 * self.sigs[t_-1] for x in range(t_-1,-1,-1)])

        self.IRtree = tree_



def irTreeToPayoffTree( Tree, get_cash_flow, get_curr_price = lambda T, p: (p, False) ):
    '''
    This is a VERY generic and powerful function, that return a payoff tree matrix with
    size = IRtree

    Parameters --
    Tree:           an interest rate Tree object
    get_cash_flow:  a function that return the cashflow at a specifc time T
    get_curr_price: a function that take two parameters (T, curr_price),
                    and decide the actual current price (incorprate early exercise).
                    the results is a tuple (acutal price, default or not)
                    By default, it's European
    '''
    ## chekc if Tree object is valid
    try:
        IRTree = Tree.IRtree
        dt = Tree.dt
        q = Tree.q
        T = Tree.T
        T_eff = int(T/dt) ## e.g. T=10, dt =0.5, T_eff =20
    except:
        raise Exception('The Tree object provided is not valid.')

    ## check tree size
    if (IRTree.shape != (T_eff, T_eff)):
        raise Exception('Invalid IRTree size={}'.format(IRTree.shape))

    payoff_tree_ = np.repeat(np.NAN, T_eff**2).reshape(T_eff, T_eff) ## Initiation
    decision_tree_ =  np.repeat(np.NAN, T_eff**2).reshape(T_eff, T_eff)
    ## The last Column case is a bit special:

    europrice_ = get_cash_flow((T_eff-1)*dt) / (1+IRTree[:,T_eff-1]*dt)
    (payoff_tree_[:,T_eff-1], decision_tree_[:,T_eff-1]) = get_curr_price((T_eff-1)*dt, europrice_)

    for t_ in range( T_eff-2, -1, -1): ## t_ = T_eff-2, T_eff-3, .., 0
        #print(t_)
        europrice_ = (get_cash_flow(t_*dt) + q * payoff_tree_[:t_+1, t_+1] \
                + (1-q) * payoff_tree_[1:t_+2, t_+1] )\
                / (1+IRTree[:t_+1, t_]*dt)
        (payoff_tree_[:t_+1,t_],decision_tree_[:t_+1,t_]) = \
            get_curr_price(t_*dt, europrice_)

    return ( payoff_tree_, decision_tree_ )


def calSpotRateDuration( IRTree, PayoffTree, i=0,j=0 ):
    '''
    get the spot rate duration at a specific node
    '''
    if(IRTree.shape != PayoffTree.shape):
        raise Exception('The trees has different size')

    return -(PayoffTree[i,j+1]-PayoffTree[i+1,j+1])/\
            (IRTree[i,j+1]-IRTree[i+1,j+1])/PayoffTree[i,j]

def calIRDelta( IRTree, PayoffTree, i=0,j=0 ):
    '''
    get the IR delta
    '''
    if(IRTree.shape != PayoffTree.shape):
        raise Exception('The trees has different size')

    return (PayoffTree[i,j+1]-PayoffTree[i+1,j+1])/\
            (IRTree[i,j+1]-IRTree[i+1,j+1])

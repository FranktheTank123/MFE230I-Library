'''
This library contains tree-based-model-related functions


Modification log:


7/7/2016 - Initiation of the script
'''

import itertools
import pandas as pd
import numpy as np
import scipy as sp
from scipy.optimize import minimize
import Fixed_Income_Library as fi
import Duration_Convexity as dc


class hoLeeTree:
    '''
    7/11/2016 -- Finalized, changed T to N to avoid confusion

    the class will construct a simple Ho-Lee tree
    this could also be (possibly) generalized into BDT tree later..
    '''
    def __init__(self, sig, N, dt = 1, q = 0.5):
        '''
        Initiation parameters -
        sig:    the constant volatility should be pre-defined
        N:      total length of trees
        dt:     each time step
        q:      the risk-neutual probability, assume to be 0.5 by default
        '''
        self.dt = dt
        self.q = q
        self.sig = sig
        self.N = N
        self.drifts = np.repeat(0.01, N) # just a place holder here...

    def getDriftPrice( self, drifts ):
        '''
        drifts: a Nx1 vector indicate the drift at each level of the tree

        will return a Nx1 price ( assume par = 1) vector, using the given drifts
        '''
        N = self.N
        sig = self.sig
        dt = self.dt
        q = self.q

        #length check
        if (len(drifts) != N):
            raise Exception("Tree size={}, while drift size={}".format(T, len(drifts)))

        prices_pred_ = np.zeros(N)


        for (ind_, time_) in  enumerate(range(1, N + 1)): # time_ = 1, 2, 3, ..., N
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
        if (len(prices) != self.N):
            raise Exception("Tree size={}, while prices size={}".format(self.N, len(prices)))


        getDriftPrice_wrapper = lambda x: ((self.getDriftPrice(x) - prices) ** 2).sum()

        results_  = minimize( getDriftPrice_wrapper, drift_init
                            , method = method, options = options)
        self.drifts = results_.x


    def plotIRTree(self):
        '''
        generate a TxT upper-triangle matrix, representing the tree of the IR
        '''
        tree_ = np.repeat(np.nan,self.N**2).reshape(self.N,self.N)
        for t_ in range(self.N):
            tree_[:t_ + 1, t_] = self.drifts[:t_ + 1].sum() + \
                        np.array([ (t_-2*x) * self.sig for x in range(t_ + 1)])
        self.IRtree = tree_

class bdtTree:
    '''
    7/11/2016 -- Finalized, changed T to N to avoid confusion

    the class will construct a BDT tree
    '''
    def __init__(self, N, dt = 1, q = 0.5):
        '''
        Initiation parameters -
        N:      total lengths of trees
        dt:     each time step
        q:      the risk-neutual probability, assume to be 0.5 by default
        '''
        self.dt = dt
        self.q = q
        self.N = N
        self.drifts = np.repeat(0.01, N)
        self.sigs = np.repeat(0, N)

    def getDriftPrice( self, drifts ):
        '''
        drifts: a Nx1 vector indicate the drift at each level of the tree
        will return a Nx1 price ( assume par = 1) vector, using the given drifts
        '''

        N = self.N
        dt = self.dt
        sigs = self.sigs
        q = self.q

        if (len(drifts) != N):
            raise Exception("drifts size={},should be {}".format(len(drifts),N))

        prices_pred_ = np.zeros(N)
        # prices_pred_ = np.zeros(T/dt)
        #print("drifts size =", len(drifts))
        #print("sigs size=",len(sigs))

        for (ind_, time_) in  enumerate(range(1, N+1)): # time_ = 1, 2, 3, ..., N
            #print(ind_, time_)
            ## the last step, temp_price_ will be a 1xn+1 vector
            temp_price_ = np.repeat( 1, time_+1)
            ## trace back the tree until it hits the first period
            while( time_ > 0):
                temp_price_ = (q*temp_price_[:-1] + (1-q)*temp_price_[1:]) / ( 1 + dt * \
                        drifts[time_-1]*np.exp([ 2* x * sigs[time_-1] * (dt**0.5) for x in range( 0 ,-time_,-1)]))
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
        #if (len(prices)+1 != self.N):
        #    raise Exception("Tree size={}, while prices size+1={}".format(self.N, len(prices)+1 ))

        #if( len(drift_init) != len(self.drifts)):
        #    raise Exception("input drift size={}, should be {}".format(len(drift_init),len(self.drifts)))

        self.sigs = np.append(0,sigs)
        r0_ = fi.zToSpot(prices[0],self.dt,n=1./self.dt)

        getDriftPrice_wrapper = lambda x, r0=r0_: ((self.getDriftPrice( np.append(r0,x)) - prices) ** 2).sum()
        getDriftPrice_wrapper1 = lambda x,: ((self.getDriftPrice( x - prices) ** 2).sum())
        results_  = minimize( getDriftPrice_wrapper, drift_init
                            , method = method, options = options)
        self.drifts = np.append(r0_,results_.x)
        #self.drifts = results_.x


    def plotIRTree(self):
        '''
        generate a N x N upper-triangle matrix, representing the tree of the IR
        '''
        N = self.N
        tree_ = np.repeat(np.nan,N**2).reshape(N,N)
        for t_ in range(1, N + 1): #t_ = 1, 2, ..., N
            tree_[:t_, t_-1] =  self.drifts[t_-1] * \
                np.exp([ 2* x * self.dt**0.5 * self.sigs[t_-1] for x in range(0,-t_,-1)])

        self.IRtree = tree_

class hullWhiteTree:
    '''
    7/11/2016 -- Finalized, changed T to N to avoid confusion

    the class will construct a HullWhite tree
    '''
    def __init__(self, N, K, sig, dt = 0.5):
        '''
        Initiation parameters -
        N:      total steps of trees
        K:      Kappa, mean reversion factor
        sig:    vol
        dt:     each time step
        '''

        self.dt = dt
        self.N = N
        self.K = K
        self.sig = sig
        self.M = (np.exp(-K*dt)-1)
        self.V = sig**2/2/K * (1-np.exp(-2*K*dt))
        self.dr = (3*self.V)**0.5
        self.j_max = int(np.ceil(-0.184/self.M))

        ## initialize the 2*j_max+1 by N IR tree
        #self.IRtree = np.repeat(np.nan,(2*self.j_max+1)*N ).reshape( 2*self.j_max+1, N )
        ## initialize the 2*j_max+1 by 3 q_table
        self.q_table = np.repeat(np.nan,(2*self.j_max+1)*3 ).reshape( 2*self.j_max+1, 3 )
        for (i,j) in enumerate(range(self.j_max,-self.j_max-1,-1)):
            if(-j>=self.j_max): ## type B
                u_temp_ = 1./6 + ((j*self.M)**2 - j*self.M )/2.
                m_temp_ = -1./3 - (j*self.M)**2 + 2*j*self.M
                d_temp_ = 7./6 + ((j*self.M)**2 - 3*j*self.M )/2.
            elif(j>=self.j_max): ## type C
                u_temp_ = 7./6 + ((j*self.M)**2 + 3*j*self.M )/2.
                m_temp_ = -1./3 - (j*self.M)**2 - 2*j*self.M
                d_temp_ = 1./6 + ((j*self.M)**2 + j*self.M )/2.
            else:
                u_temp_ = 1./6 + ((j*self.M)**2 + j*self.M )/2.
                m_temp_ = 2./3 - (j*self.M)**2
                d_temp_ = 1./6 + ((j*self.M)**2 - j*self.M )/2.
            self.q_table[i] = [u_temp_,m_temp_,d_temp_]

        self.drifts = np.repeat(0.1, self.N)


    def getDriftPrice( self, drifts_input ):
        '''
        ##para: a (T+T-1)x1 vector indicate the drift&sig at each level of the tree
        ##will return a Tx1 price ( assume par = 1) vector, using the given drifts
        '''
        drifts = drifts_input
        N = self.N
        dt = self.dt
        sig = self.sig
        q_table = self.q_table
        j_max = self.j_max
        dr = self.dr

        if (len(drifts) != N):
            raise Exception("drifts size={},should be {}".format(len(drifts), N))


        prices_pred_ = np.zeros(N)

        #print("drifts size =", len(drifts))
        #print("sigs size=",len(sigs))

        for (ind_, time_) in  enumerate(range(1, N + 1)): # time_ = 1, 2, 3, ..., N
            #print(ind_, time_)
            ## the last step, temp_price_ will be a 1xn+1 vector
            #print (time_)
            if(time_ >= j_max):
                temp_price_ = np.repeat( 1., 2*j_max+1)
                reduce_ = 1
            else:
                temp_price_ = np.repeat( 1., 2*time_+1)
                reduce_ = 5-time_+1

            ## trace back the tree until it hits the first period
            while( time_ > j_max): ## the case when still 2*j_max+1 nodes
                new_price_ = np.repeat( 1., 2*j_max+1) ## initialization
                new_price_[0] = (q_table[0,:] * temp_price_[:3]).sum()
                new_price_[-1] =  (q_table[-1,:] * temp_price_[-3:]).sum()
                for i in range(1, 2*j_max): ## optimize later
                    #print(i,2*j_max+1,temp_price_[i-1:i+2])
                    new_price_[i] = (q_table[i,:] * temp_price_[i-1:i+2]).sum()

                #temp_price_ = new_price_ * np.exp(-(drifts[:time_].sum()+ dr*np.arange(j_max,-j_max-1,-1))*dt)
                temp_price_ = new_price_ / np.exp(drifts[:time_].sum()+ dr*np.arange(j_max,-j_max-1,-1))
                time_ -= 1


            while( time_>0): ## now nodes decrease 2 per steps
                #print ("\n",time_,temp_price_)
                new_price_ = temp_price_[1:-1]
                for (i,_) in enumerate(new_price_):
                    #print(i, new_price_)
                    new_price_[i] = (q_table[reduce_+i,:] * temp_price_[i:i+3]).sum()
                #temp_price_ = new_price_ * np.exp(-(drifts[:time_].sum()+ dr*np.arange(j_max-reduce_,-j_max+reduce_-1,-1))*dt)
                temp_price_ = new_price_ / np.exp(drifts[:time_].sum()+ dr*np.arange(j_max-reduce_,-j_max+reduce_-1,-1))
                reduce_ += 1
                time_ -= 1
            ## append the results
            prices_pred_[ ind_] = temp_price_[0]

        return prices_pred_


    def calibrateDrift(self, prices, drift_init, method = 'nelder-mead'
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
        if (len(prices) != self.N):
            raise Exception("Tree size={}, while prices size={}".format(self.N, len(prices)))
        #if( len(drift_init) != len(self.drifts)-1):
        #    raise Exception("input drift size={}, should be {}".format(len(drift_init),len(self.drifts)-1))

        r0_ = -np.log(prices[0])
        #r0_ = -np.log(prices[0])/self.dt
        ## a wrapper
        getDriftPrice_wrapper = lambda x, r0 = r0_: ((self.getDriftPrice(np.append(r0,x)) - prices) ** 2).sum()
        getDriftPrice_wrapper1 = lambda x : ((self.getDriftPrice(x) - prices) ** 2).sum()
        results_  = minimize( getDriftPrice_wrapper1, drift_init
                            , method = method, options = options)

        #self.drifts = np.append(r0_,results_.x)
        self.drifts = results_.x


    def plotIRTree(self):
        '''
        generate a N x N upper-triangle matrix, representing the tree of the IR
        '''
        N = self.N
        dt = self.dt
        sig = self.sig
        q_table = self.q_table
        j_max = self.j_max
        dr = self.dr
        drifts = self.drifts

        tree_ = np.repeat(np.nan,(2*j_max+1)*N ).reshape( 2*j_max+1, N )

        reduce_ = 1
        for i in range(N-1, -1, -1):
            if(i >= j_max):
                tree_[:,i] = drifts[:i+1].sum()+ dr * np.arange(j_max,-j_max-1,-1)
                i -= 1
            else:
                #print(reduce_)
                tree_[reduce_:-reduce_,i] = drifts[:i+1].sum()+ dr * np.arange(j_max-reduce_,-j_max+reduce_-1,-1)
                i -= 1
                reduce_ += 1

        self.IRtree = tree_

def irTreeToPayoffTree( Tree, get_cash_flow, get_curr_price = lambda T, p: (p, False) ):
    '''
    This is a VERY generic and powerful function, that return a payoff tree matrix with
    size = IRtree

    Parameters --
    Tree:           an interest rate Tree object, size of IR tree should be NxN
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
        N = Tree.N
        #T_eff =  ## e.g. T=10, dt =0.5, T_eff =20
    except:
        raise Exception('The Tree object provided is not valid.')

    ## check tree size
    if (IRTree.shape != (N, N)):
        raise Exception('Invalid IRTree size={}'.format(IRTree.shape))


    payoff_tree_ = np.repeat(np.NAN, N**2).reshape(N, N) ## Initiation
    decision_tree_ =  np.repeat(np.NAN, N**2).reshape(N, N)

    ## The last Column case is a bit special:
    europrice_ = get_cash_flow((N-1)*dt) / (1+IRTree[:,N-1]*dt)
    (payoff_tree_[:,N-1], decision_tree_[:,N-1]) = get_curr_price((N-1)*dt, europrice_)

    for t_ in range( N-2, -1, -1): ## t_ = N-2, N-3, .., 0
        #print(t_)
        europrice_ = (get_cash_flow(t_*dt) + q * payoff_tree_[:t_+1, t_+1] \
                + (1-q) * payoff_tree_[1:t_+2, t_+1] )\
                / (1+IRTree[:t_+1, t_]*dt)
        (payoff_tree_[:t_+1,t_],decision_tree_[:t_+1,t_]) = \
            get_curr_price(t_*dt, europrice_)

    return ( payoff_tree_, decision_tree_ )

def mbsGetPrepayRate(curr_rate,interest_rate):
    '''
    Linear interpolation of the prepayment rate

    if curr_rate - interest rate <= 0  --- 3%
    if curr_rate - interest rate in [0,50bp] --- 3% to 5%
    if curr_rate - interest rate in [50bp,100bp] --- 5% to 8%
    if curr_rate - interest rate in [100bp,200bp] --- 8% to 17%
    '''
    x_ = curr_rate - interest_rate

    if (np.isnan(x_)):
        return np.nan
    elif ( x_ <= 0):
        return 0.03
    elif ( x_ <= 0.005):
        return 0.03 + (x_- 0)/(0.005)* 0.02
    elif (x_ <=0.01):
        return 0.05 + (x_-0.005)/0.005* 0.03
    elif (x_ <= 0.02):
        return 0.08 + (x_-0.01)/0.01 * 0.09
    else:
        return 0.17

def mbsPricer(path_array, HWT, MBS_quote_rate, init_par = 1000000, \
                get_prepay_rate = lambda curr_rate,interest_rate: 0):
    '''
    Parameters:
    path_array: arrays of -1, 0, 1's, should be with length N-1 (as the first point is fixed,
                no need to randomize)
    HWT:        a HullWhite tree
    MBS_interest:   the corresponding rate of MBS, with the same discounting measure as in HWT.IRtree
    init_par:   initial par value of the MBS
    get_prepay_rate: lambda curr_rate,interest_rate: 0 (i.e, by default, no prepayment allowed)
    '''

    MBS_interest = np.log(1+MBS_quote_rate/2)
    N = len(path_array)+1
    IRTree = HWT.IRtree[:,:N+1] ## only need the first N columns in this case
    j_max = HWT.j_max
    location_array_ = np.repeat(j_max,N) ## initialized the array
    q_table = HWT.q_table

    for (i,j) in enumerate(path_array):
        if location_array_[i] == 2*j_max:
            location_array_[i+1] = location_array_[i] + j-1
        elif location_array_[i] == 0:
            location_array_[i+1] = location_array_[i] + j+1
        else:
            location_array_[i+1] = location_array_[i] + j

    ## discout array given path Nx1
    discount_array_ = np.array([ IRTree[j,i] for (i,j) in enumerate(location_array_)])
    ## prepayrate array given path Nx1
    prepayrate_array_ = np.array([ get_prepay_rate(discount_array_[i], MBS_interest) \
                                  for (i,j) in enumerate(location_array_)])
    prepayrate_array_[0] = 0 ## no prepayment at time 0

    ## probability array given path (N-1)x1
    #print(location_array_)
    prob_path_ = np.array([ q_table[location_array_[i],j+1] for (i,j) in enumerate(path_array)] )
    ## total probability of array of realization
    total_prob_ = np.prod(prob_path_)

    ## we assume the prepayment is from t=0,..,4.5, and annuity payment is from t=0.5,...5
    dt=HWT.dt
    prepayment = np.zeros(N+1)
    annuity_payment = np.zeros(N+1)
    principle_payment = np.zeros(N+1)

    total_payment = np.zeros(N+1) ## from t=0,...,5
    curr_par = init_par


    for i in range(N):
        prepayment[i] = curr_par*prepayrate_array_[i] ## this is paid at year i*dt, e.g., 0, 0.5, 1,..,4.5

        curr_par -= prepayment[i] ## we deduct out prepayment from the par
        (temp1_, temp2_, _) = fi.calAnnuityPayment(5-i*dt, MBS_quote_rate, n =2, par= curr_par)
        annuity_payment[i+1] = temp1_
        principle_payment[i+1] = temp2_[0]
        curr_par -= principle_payment[i+1] ## then we deduct how much principle we are about to pay next period
        #print(curr_par, prepayment[i])

    total_payment =  prepayment + annuity_payment
    ## now let's put the sum together with discount
    sum_total_payment = (total_payment[1:]/np.exp(np.cumsum(discount_array_))).sum()
    return [sum_total_payment, total_prob_]

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

def path_gen(length = 9, vals = np.array([-1,0,1]), random = False, path = 1000\
            , antithetic = False):
    '''
    Path generator:

    length:   length of the array size
    vals:     possible values of each path element
    random:   if = F, will return all possible paths
    path:     when random = T, generate that many paths
    antithetic: will return antithetic paths as well (2*path paths), assume array is symmetric...
    '''
    val_size_ = len(vals)

    assert (val_size_ and length and path), "you don't want any paths!?"

    all_path = np.array(list(itertools.product(vals,repeat=length)))

    if(not random):
        ## this gives all possibility of path combinations at each step...
        return all_path

    ## this gives path x length matrix
    #random_samples_ = np.array([[vals[int(i)] for i in np.floor(np.random.rand(length)*val_size_)]\
    #        for x in range(path)])
    random_samples_ = np.array([all_path[np.random.randint(0,val_size_**length)] for x in range(path)])


    if(not antithetic) : ## random, not antithetic
        return random_samples_
    else: ## random, antithetic
        return np.concatenate((random_samples_, -random_samples_), axis=0)## lazy evaluation...

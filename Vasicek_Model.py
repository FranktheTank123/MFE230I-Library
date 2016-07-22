'''
This library contains tree-based-model-related functions


Modification log:


7/15/2016 - Initiation of the script
'''

import pandas as pd
import numpy as np
import scipy as sp
from scipy.stats import norm
from scipy.optimize import minimize, fmin
import Fixed_Income_Library as fi
from numpy import linalg


def getVasicekZ(kappa, mu, sig, t, r_t):
    '''
    calculate the discount rate of a Vasicek model given r_t and T
    '''
    return np.exp( -(mu * t + (1-np.exp(-kappa*t))*( r_t - mu ) / kappa \
         - sig**2/2./kappa**2 * ( t + (1-np.exp(-2*kappa*t)) /2./kappa \
         - 2 * (1 -np.exp(-kappa*t)) / kappa)))

def vasicekMean(t,s,kappa, mu, r_s):
    '''
    Calculate the conditional mean of the vasicek model
    '''
    return r_s*np.exp(-kappa*(t-s))+mu*(1-np.exp(-kappa*(t-s)))

def vasicekVar(t,s,kappa, mu, sig, r_s):
    '''
    Get the conditional variance of the vasicek model
    '''
    return sig**2/2/kappa *(1-np.exp(-2*kappa*(t-s)))

def vasicekZCBOption(TB,TO,F,K,r,mu,kappa,sigma, getZ_):
    '''
    Option Price of Vasic Model for zero coupon bond
    Here F is the face value of the bond
    K is the strike price of the bond

    '''
    #print(TB,TO,F,K,r,mu,kappa,sigma)
    k = K/F
    ZrtTB = getZ_(r, TB)
    ZrtTO = getZ_(r, TO)
    term1 = ( 1 - np.exp( -2*kappa * TO ) ) * sigma**2 / (2*kappa)
    term2 = (1 - np.exp( -kappa * (TB-TO) ) )**2 / (kappa**2)
    sigmaz = np.sqrt( term1*term2 )
    h = np.log(ZrtTB/(k*ZrtTO)) / sigmaz + sigmaz/2
    Nh = norm.cdf(h,loc=0,scale = 1)
    Nh_sgimaz = norm.cdf(h-sigmaz,loc=0,scale=1)
    C = ZrtTB*Nh - k*ZrtTO*Nh_sgimaz
    #print('ZTB',ZrtTB,'ZTO',ZrtTO,'h:',h)
    return (C*F)

def euroBondOptionPricer_optimizer(r_t, t, T, c, kappa, mu, par,dt,getZ_):
    '''
    this function gives the RMSE between the par value and the expected PV of
    a coupon bond with spot rate = r_t
    '''
    ## get back the remaining pay time
    ## e.g. t= 0.5, T=3, dt = 1 -> pay_time_ = [0.5, 1.5, 2.5]
    ## e.g. t = 1, T= 4, dt = 0.5 -> pay_time_ = [0.5,1, ..., 3]
    pay_time_ = np.array([x+dt-t for x in np.arange(0,T,dt)])
    pay_time_ = pay_time_[pay_time_>0] ## filter  is not needed

    ## get the cash flow
    cash_flow_ = np.repeat(c*dt*par, len(pay_time_))
    cash_flow_[-1] += par ## add the par

    ## get the discount factors
    disc_factors_ = np.array([getZ_(r_t,t_)[0] for t_ in pay_time_])

    ## the the PV of the sum of the cash flow
    PV_t_cash_flow_ = np.dot(cash_flow_ , disc_factors_)

    ## return the cost function
    ## print (PV_t_cash_flow_,cash_flow_,disc_factors_ )
    return ((PV_t_cash_flow_-par)**2)




def euroBondOptionPricer(t, T, c, r_0, kappa, mu,sig, getZ_, par = 100, dt = 0.5, bump = 1e-5):
    '''
    price the value of an European bond option using Vasicek model
    possible to be generalized to other models...
    '''
    ## we need to first find out the strike r_t of exercising the option or not
    strike_r_ = fmin(euroBondOptionPricer_optimizer, \
                       r_0, args=(t,T,c,kappa,mu,par,dt,getZ_),xtol=10e-7)[0]
    print('strike r =',strike_r_)
    ## get back the remaining pay time
    ## e.g. t= 0.5, T=3, dt = 1 -> pay_time_ = [0.5, 1.5, 2.5]
    ## e.g. t = 1, T= 4, dt = 0.5 -> pay_time_ = [0.5,1, ..., 3]
    pay_time_ = np.array([x+dt-t for x in np.arange(0,T,dt)])
    pay_time_ = pay_time_[pay_time_>0] ## filter  is not needed

    ## get the cash flow
    cash_flow_ = np.repeat(c* dt *par, len(pay_time_))
    cash_flow_[-1] += par

    ## get the discount factors for K
    disc_factors_ = np.array([getZ_(strike_r_, t_) for t_ in pay_time_])

    ## get the prices of each sub ZCB when r_t = strike rate
    strike_prices_ = cash_flow_ * disc_factors_

    ## get the call price of each sub ZCB
    call_prices_ = np.array([vasicekZCBOption(T_B,t, CF, K, r_0, mu, kappa, sig, getZ_) \
                             for (K, T_B, CF) in zip(strike_prices_, pay_time_+t, cash_flow_)])
    ## return the entire call bond price, also the sub ZCB calls, all sub call deltas, and all ZCB deltas
    #return (call_prices_.sum(),call_prices_, call_deltas_, ZCB_deltas_)
    return call_prices_.sum()

def markovChain(dt, n):
    tran_mat_ = np.array([[1-0.1*dt, 0.1*dt, 0],[0, 1-0.2*dt, 0.2*dt],[0.5*dt,0, 1-0.5*dt]])
    return linalg.matrix_power(tran_mat_,n)

def MarkovR(i, r, P, dt, T):
    '''
    parameters:
    i:    the index of r
    r:    [0,5%,10%]
    '''
    Nmax = int(T/dt)
    ## initialize the yield factor
    y = np.zeros(Nmax)
    y[0] = r[i]

    price_curr = np.zeros(3) + np.exp(r[i] * dt)
    prob_curr = P[i,:]

    for n in np.arange(1, Nmax):
        price_tmp = price_curr * np.exp(r*dt)
        y[n] = np.log( np.sum(prob_curr * price_tmp) ) / dt / (n + 1)
        for j in range(3):
            prob_tmp = prob_curr * P[:,j]
            price_curr[j] = price_tmp[0] * prob_tmp[0] + \
                     price_tmp[1] * prob_tmp[1] + \
                     price_tmp[2] * prob_tmp[2]
            price_curr[j] = price_curr[j] / np.sum(prob_tmp)
        prob_curr = P.T.dot(prob_curr)
    return y

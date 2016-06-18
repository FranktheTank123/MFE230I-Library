'''
This library contains duration- and convexity-related functions

Modification log:

6/17/2016 - Initiation of the script. I think it would be benefitial to have a bond
class to deal with different features OOP seems to be helpful in this case

'''

import pandas as pd
import numpy as np
import scipy as sp
import Fixed_Income_Library as fi
fi = reload(fi)



'''
the market object should properly return the Z and spot rate for any T

To initiate this object, you should pass in a get_Z(T) function, or you can pass
in a get_spot(T, n) function

WARNING 6/18/2016: it's equivalent to use Z and semi-annual spot rate
'''

class market:
    def __init__(self, function, fun_type = 'Z'):
        if (fun_type == 'Z'):
            self.get_Z = lambda T : function( T )
            self.get_spot = lambda T, n = 2, t = 0: fi.zToSpot( self.get_Z(T), T, t = t, n = n)
        elif(fun_type == 'S'):
            self.get_spot = lambda T, n = 2 : function( T, n )
            self.get_Z = lambda T, n = 2, t = 0 : fi.spotToZ(self.get_spot(T), T, t = t, n = n )
        else:
            raise Exception('unkown input function type')


'''
The bond class.
'''
class bond:
    def __init__(self, maturity, price = 100, par = 100, coupon = 0, freq = 2):
        self.price = price
        self.coupon = coupon # this is dollar coupon
        self.maturity = maturity
        self.freq = freq
        self.par = par
        self.payment_count = np.ceil( freq * self.maturity ) # how may coupon payment do we pay
                                                 # we use the ceiling function to make sure the coupon
                                                 # is always paid at T

        '''
        note: coupon_date is in descending order e.g. [2, 1, 1.5, 0.5] (0 does not count)
        '''
        self.coupon_date = np.array(self.maturity - (np.arange(1, self.payment_count+1)-1)/2.0)

    # we can backup the coupon rate using the market curve and price
    '''
    note: the coupon rate is quoted annually
    '''
    def set_coupon(self, market):
        self.coupon = (self.freq * (self.price - self.par * market.get_Z( self.coupon_date[0] ) ) /
                        market.get_Z( self.coupon_date ).sum())[0]### note: [0] is that we want coupon to be a number

    def set_price(self, market):
        self.price = (self.coupon/self.freq * market.get_Z( self.coupon_date ).sum() + self.par * market.get_Z( self.coupon_date[0]))[0]

    # since YTM varies with the market, we need the current market curve
    def set_YTM(self, market, isCts = False ):
        if(not isCts): # if you want discrete YTM
            fun_ = lambda y : (self.coupon / self.freq *
                    np.power(( 1 + y / self.freq ), -self.coupon_date * self.freq).sum()
                    + self.par * np.power(( 1 + y / self.freq ), -self.coupon_date[0] * self.freq) - self.price)
        else:   # if you want cts YTM
            fun_ = lambda y : (self.coupon / self.freq *
                    np.exp( - y * self.coupon_date).sum()
                    + self.par * np.exp( - y * self.coupon_date[0])- self.price)

        self.YTM = sp.optimize.newton( fun_ , self.coupon/self.par )
        self.isCtsYTM = isCts # this will be important later when we calculate the duration

    '''
    set the 3 durations (Mac, Mod, DV01) and convexity
    '''
    def set_dc(self):
        if( self.isCtsYTM ): # when YTM is continuous
            self.Mac_duration =  ((self.coupon_date * self.coupon / self.freq *
                                    np.exp(-self.YTM * self.coupon_date)).sum() / self.price
                                    + self.coupon_date[0] * self.par * np.exp(-self.YTM * self.coupon_date[0]) / self.price )
            self.Mod_duration = self.Mac_duration
            self.convextiy = ( (np.power(self.coupon_date,2) * self.coupon / self.freq *
                                np.exp(-self.YTM * self.coupon_date) ).sum() + np.power(self.coupon_date[0],2)*self.par * np.exp(-self.YTM * self.coupon_date[0])
                                )/ self.price

        else: # when YTM is NOT continuous
            self.Mac_duration =  ((self.coupon_date * self.coupon / self.freq *
                                    np.power( 1 + self.YTM/self.freq , -self.freq * self.coupon_date)).sum() / self.price
                                    + self.coupon_date[0] * self.par * np.power( 1 + self.YTM/self.freq , -self.freq * self.coupon_date[0]) / self.price )
            self.Mod_duration = self.Mac_duration / (1 + self.YTM/self.freq)
            self.convextiy =  ((self.coupon/self.freq * (self.coupon_date * (self.coupon_date+1/self.freq) *
                              np.power(1+self.YTM/self.freq, -self.freq* self.coupon_date)).sum()
                              + self.par * self.coupon_date[0] * (self.coupon_date[0]+1/self.freq) *np.power(1+self.YTM/self.freq, -self.freq* self.coupon_date[0]) )
                              * np.power( 1 + self.YTM /self.freq , -2 ) / self.price)
        # DV 01 is uniform
        self.DV01 = self.Mod_duration * self.price / 10000

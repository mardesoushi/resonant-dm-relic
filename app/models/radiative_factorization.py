import numpy as np
from  models.general_parameters import *
alph = (1/137)    # QED coupling constant 
from scipy.integrate import quad
#import fortran_quad

import scipy.special as sp
from sympy import *

################ ISR
from scipy.integrate import quad
erroabs = 1
errorel = 1
intlimit = 10

@np.vectorize
def HPhsig(s, mq, Mmed, mx, gr, gl, gx, sig0, Nf):
    
    #xgmin = 0.10 ## 4% do feixe
    #xgmax = 0.66 ## 66% do feixe
    
    # Atualização 24-06-2023
    #xgmin = (2 / np.sqrt(s)) * 0.060  ## q0 = 60 GeV
    xgmin = (2 / np.sqrt(s)) * 0.060
    #xgmin = 0.084 ## 4% do feixe
    xgmax = (0.20 * Mmed) / np.sqrt(s)
    ### CLIC CONSTRAINTS ####
    # A = 0.060 ## E_min = 0.060 TeV
    # xgmin = (2 * A) / np.sqrt(s)
    # qmax = 1.0 ## E_max = 1 TeV
    # xgmax = (2 * qmax) / np.sqrt(s)
    

    @np.vectorize
    def integrand_xg(xg, s, mq, Mmed, mx, gr, gl, gx, sig0):

        preterm   = 1 / xg
        integrand = (1 - (xg) + ((xg**2) / 2))
        sigcorr   = sig0(s*(1-xg), mq, Mmed, mx, gr, gl, gx, Nf) / sig0(s, mq, Mmed, mx, gr, gl, gx, Nf)
        result    = preterm * integrand * sigcorr
        return result

    def i_argument(x):
        result = integrand_xg(x, s, mq, Mmed, mx, gr, gl, gx, sig0)
        return result
    
    #integral = quad(integrand_xg, xgmin, xgmax, args=(s, mq, Mmed, mx, gr, gl, gx, sig0), limit=intlimit, epsabs=erroabs, epsrel=errorel)[0]      
    integral = fortran_quad.quad16(i_argument, xgmin, xgmax)

    parentesis1  = (-1 + 2*np.log10(np.sqrt(s)/mq))         ## sempre positivo
    parentesis2  = (np.log10(xgmin) + (13/12) + integral)   ## positivo para xgmin > 0.083, a integral é sempre positiva
    termcte      = ((-17/36) + (1/6)*(np.pi**2))            ## sempre positivo, = 1.17271184
    keys_tot     = (parentesis1 * parentesis2)  + termcte   ## positivo se parentesis2 for positivo
    delta        = ((2*alph/np.pi)*keys_tot)
    hp           = sig0(s, mq, Mmed, mx, gr, gl, gx, Nf) * (1 + delta )  ## xsec sempre positiva
    return hp





# ## Implementation of: arXiv:1912.02870v2 and Gondolo 1991

# erroabs = 1
# errorel = 1
# intlimit = 10

# ### CLIC CONSTRAINTS ####
# # A = 0.060 ## E_min = 0.060 TeV
# # xgmin = (2 * A) / np.sqrt(s)
# # qmax = 1.0 ## E_max = 1 TeV
# # xgmax = (2 * qmax) / np.sqrt(s)

# #xgmin = 0.10 ## 4% do feixe
# #xgmax = 0.66 ## 66% do feixe


@np.vectorize
def integrandxi(xg, s, mq, Mmed, mx, gr, gl, gx, sig0, Nf):


    # Atualização 24-06-2023
    #xgmin = (2 / np.sqrt(s)) * 0.060  ## q0 = 60 GeV
    xgmin = (2 / np.sqrt(s)) * 0.060
    #xgmin = 0.084 ## 4% do feixe
    xgmax = (0.20 * Mmed) / np.sqrt(s)
    ### CLIC CONSTRAINTS ####
    # A = 0.060 ## E_min = 0.060 TeV
    # xgmin = (2 * A) / np.sqrt(s)
    # qmax = 1.0 ## E_max = 1 TeV
    # xgmax = (2 * qmax) / np.sqrt(s)

    
    @np.vectorize
    def integrand_xg(xg, s, mq, Mmed, mx, gr, gl, gx, sig0):

        preterm   = 1 / xg
        integrand = (1 - (xg) + ((xg**2) / 2))
        sigcorr   = sig0(s*(1-xg), mq, Mmed, mx, gr, gl, gx, Nf) / sig0(s, mq, Mmed, mx, gr, gl, gx, Nf)
        result    = preterm * integrand * sigcorr
        return result

    # def i_argument(x):
    #     result = integrand_xg(x, s, mq, Mmed, mx, gr, gl, gx, sig0)
    #     return result
    
    #integral = quad(integrand_xg, xgmin, xgmax, args=(s, mq, Mmed, mx, gr, gl, gx, sig0), limit=intlimit, epsabs=erroabs, epsrel=errorel)[0] ## x1 integration!! - The result is tau*\integral{f(x)} (mq is set to 0 because python errors in this phase)      
    #integral = fortran_quad.quad16(i_argument, xgmin, xgmax)
    integral = integrand_xg(xg, s, mq, Mmed, mx, gr, gl, gx, sig0)
    
    # parentesis1 = (-1 + 2*np.log10(np.sqrt(s)/mq))
    # parentesis2  = (np.log10(xgmin) + (13/12) + integral)
    # keys_tot = (parentesis1 * parentesis2) + ((-17/36) + (1/6)*(np.pi**2))
    # hp =  sig0(s, mq, Mmed, mx, gr, gl, gx) * (1 + ((2*alph/np.pi)*keys_tot) )
    return integral


@np.vectorize
def deltaonly(xgm, xgmx, s, mq, Mmed, mx, gr, gl, gx, sig0, Nf):



    # Atualização 24-06-2023
    #xgmin = (2 / np.sqrt(s)) * 0.060  ## q0 = 60 GeV
    xgmin = (2 / np.sqrt(s)) * 0.060
    #xgmin = 0.084 ## 4% do feixe
    xgmax = (0.20 * Mmed) / np.sqrt(s)
    ### CLIC CONSTRAINTS ####
    # A = 0.060 ## E_min = 0.060 TeV
    # xgmin = (2 * A) / np.sqrt(s)
    # qmax = 1.0 ## E_max = 1 TeV
    # xgmax = (2 * qmax) / np.sqrt(s)
    @np.vectorize
    def integrand_xg(xg, s, mq, Mmed, mx, gr, gl, gx, sig0):

        preterm   = 1 / xg
        integrand = (1 - (xg) + ((xg**2) / 2))
        sigcorr   = sig0(s*(1-xg), mq, Mmed, mx, gr, gl, gx, Nf) / sig0(s, mq, Mmed, mx, gr, gl, gx, Nf)
        result    = preterm * integrand * sigcorr
        return result

    ## Função para chamar a integral em termos de uma variável só
    def i_argument(x):
        result = integrand_xg(x, s, mq, Mmed, mx, gr, gl, gx, sig0)
        return result
    integral = fortran_quad.quad16(i_argument, xgm, xgmx)
    #integral = integrand_xg(xg, s, mq, Mmed, mx, gr, gl, gx, sig0)

    parentesis1  = (-1 + 2*np.log10(np.sqrt(s)/mq))         ## sempre positivo
    parentesis2  = (np.log10(xgm) + (13/12) + integral)   ## positivo para xgmin > 0.083, a integral é sempre positiva
    termcte      = ((-17/36) + (1/6)*(np.pi**2))            ## sempre positivo, = 1.17271184
    keys_tot     = (parentesis1 * parentesis2)  + termcte   ## positivo se parentesis2 for positivo
    delta        = ((2*alph/np.pi)*keys_tot)
    hp           = (1 + delta ) #* sig0(s, mq, Mmed, mx, gr, gl, gx)  ## xsec sempre positiva
    return hp


@np.vectorize
def dsigmadxgamma(xg, s, mq, Mmed, mx, gr, gl, gx, sig0, Nf):


    # Atualização 24-06-2023
    #xgmin = (2 / np.sqrt(s)) * 0.060  ## q0 = 60 GeV
    xgmin = (2 / np.sqrt(s)) * 0.060
    #xgmin = 0.084 ## 4% do feixe
    xgmax = (0.20 * Mmed) / np.sqrt(s)
    ### CLIC CONSTRAINTS ####
    # A = 0.060 ## E_min = 0.060 TeV
    # xgmin = (2 * A) / np.sqrt(s)
    # qmax = 1.0 ## E_max = 1 TeV
    # xgmax = (2 * qmax) / np.sqrt(s)

    @np.vectorize
    def integrand_xg(xg, s, mq, Mmed, mx, gr, gl, gx, sig0):

        preterm   = 1 / xg
        integrand = (1 - (xg) + ((xg**2) / 2))
        sigcorr   = sig0(s*(1-xg), mq, Mmed, mx, gr, gl, gx, Nf) / sig0(s, mq, Mmed, mx, gr, gl, gx, Nf)
        result    = preterm * integrand * sigcorr
        return result

    ## Função para chamar a integral em termos de uma variável só
    # def i_argument(x):
    #     result = integrand_xg(x, s, mq, Mmed, mx, gr, gl, gx, sig0)
    #     return result
    #integral = fortran_quad.quad16(i_argument, xgmin, xgmax)
    integral = integrand_xg(xg, s, mq, Mmed, mx, gr, gl, gx, sig0)

    parentesis1  = (-1 + 2*np.log10(np.sqrt(s)/mq))         ## sempre positivo
    parentesis2  = (np.log10(xgmin) + (13/12) + integral)   ## positivo para xgmin > 0.083, a integral é sempre positiva
    termcte      = ((-17/36) + (1/6)*(np.pi**2))            ## sempre positivo, = 1.17271184
    keys_tot     = (parentesis1 * parentesis2)  + termcte   ## positivo se parentesis2 for positivo
    delta        = ((2*alph/np.pi)*keys_tot)
    hp           = sig0(s, mq, Mmed, mx, gr, gl, gx, Nf) * (1 + delta )  ## xsec sempre positiva
    return hp































### - NOT USED - ###

#Original
#a = 0.010
#b = 0.100

# xgmin = 0.083 ## 4% do feixe
# xgmax = 0.66 ## 66% do feixe

# @np.vectorize
# def dsigdgammaHP(xgmin, xg, s, mq, Mmed, mx, gr, gl, gx, sig0):
#     @np.vectorize
#     def integrand_xg(xg, s, mq, Mmed, mx, gr, gl, gx, sig0):
#         q0 = xg*np.sqrt(s)/2
#         a = 0.04

#         return ((1/q0) * (1 - (q0 /((np.sqrt(s)/2))) + ((q0**2) / (s/2) ) ))* (sig0(xg, s*(1-xg), mq, Mmed, mx, gr, gl, gx)/sig0(xg, s, mq, Mmed, mx, gr, gl, gx) )

#     a = xgmin
#     #b = 0.100

#     #INTEGRAL = quad(integrand_xg, a, b, args=(s, mq, Mmed, mx, gr, gl, gx, sig0), limit=intlimit, epsabs=erroabs, epsrel=errorel)[0] ## x1 integration!! - The result is tau*\integral{f(x)} (mq is set to 0 because python errors in this phase)      
    
#     INTEGRAL = integrand_xg(xg, s, mq, Mmed, mx, gr, gl, gx, sig0)

#     A = xgmin*np.sqrt(s)/2 # qmin
#     hp = sig0(xg, s, mq, Mmed, mx, gr, gl, gx) * (1 + (2*alph/np.pi) * ( ((-1 + 2*np.log10( np.sqrt(s)/mq)) *  ( np.log10(A/(np.sqrt(s)/2)) + 13/12 + INTEGRAL)) - 17/36 + ((1/6)*(np.pi**2))    ))

#     return hp       


# @np.vectorize
# def dsigdgammaHP_deltaonly(xg, s, mq, Mmed, mx, gr, gl, gx, sig0):
#     @np.vectorize
#     def integrand_xg(xg, s, mq, Mmed, mx, gr, gl, gx, sig0):
#         q0 = xg*np.sqrt(s)/2
#         A = np.min(q0)  ## q_min
#         return ((1/q0) * (1 - (q0 /((np.sqrt(s)/2))) + ((q0**2) / (s/2) ) ))* (sig0(xg, s*(1-xg), mq, Mmed, mx, gr, gl, gx)/sig0(xg, s, mq, Mmed, mx, gr, gl, gx) )

#     a = 0.001
#     b = 0.100

#     #INTEGRAL = quad(integrand_xg, a, b, args=(s, mq, Mmed, mx, gr, gl, gx, sig0), limit=intlimit, epsabs=erroabs, epsrel=errorel)[0] ## x1 integration!! - The result is tau*\integral{f(x)} (mq is set to 0 because python errors in this phase)      
    
#     INTEGRAL = integrand_xg(xg, s, mq, Mmed, mx, gr, gl, gx, sig0)

#     A = a*np.sqrt(s)/2 # qmin
#     hp = (1 + (2*alph/np.pi) * ( ((-1 + 2*np.log10( np.sqrt(s)/mq)) *  ( np.log10(A/(np.sqrt(s)/2)) + 13/12 + INTEGRAL)) - 17/36 + ((1/6)*(np.pi**2))    ))
#     #hp =  integrand_xg(xg, s, mq, Mmed, mx, gr, gl, gx, sig0)
#     return hp       


# @np.vectorize
# def dsigdgammaHP_deltaonly2(xg, s, mq, Mmed, mx, gr, gl, gx, sig0):
#     @np.vectorize
#     def integrand_xg(xg, s, mq, Mmed, mx, gr, gl, gx, sig0):
#         q0 = xg*np.sqrt(s)/2
#         #print('q0', q0)
#         A = np.min(q0)  ## q_min
#         result = ((1/q0) * (1 - (q0 /((np.sqrt(s)/2))) + ((q0**2) / (s/2) ) ))* (sig0(xg, s*(1-xg), mq, Mmed, mx, gr, gl, gx)/sig0(xg, s, mq, Mmed, mx, gr, gl, gx) )
#         return result

#     a = 0.04
#     b = 0.66
#     #a = 0.001
#     #b = 0.100

#     #INTEGRAL = quad(integrand_xg, a, b, args=(s, mq, Mmed, mx, gr, gl, gx, sig0), limit=intlimit, epsabs=erroabs, epsrel=errorel)[0] ## x1 integration!! - The result is tau*\integral{f(x)} (mq is set to 0 because python errors in this phase)      
    
#     INTEGRAL = integrand_xg(xg, s, mq, Mmed, mx, gr, gl, gx, sig0)

#     A = a*np.sqrt(s)/2 # qmin
#     hp = (1 + (2*alph/np.pi) * ( ((-1 + 2*np.log10( np.sqrt(s)/mq)) *  ( np.log10(A/(np.sqrt(s)/2)) + 13/12 + INTEGRAL)) - 17/36 + ((1/6)*(np.pi**2))    ))
#     #hp =  integrand_xg(xg, s, mq, Mmed, mx, gr, gl, gx, sig0)
#     return hp     


# @np.vectorize
# def integrand_xi(xg, s, mq, Mmed, mx, gr, gl, gx, sig0):
#     @np.vectorize
#     def integrand_xg(xg, s, mq, Mmed, mx, gr, gl, gx, sig0):
#         q0 = xg*np.sqrt(s)/2
#         #print('q0', q0)
#         A = np.min(q0)  ## q_min
#         result = ((1/q0) * (1 - (q0 /((np.sqrt(s)/2))) + ((q0**2) / (s/2) ) ))* (sig0(xg, s*(1-xg), mq, Mmed, mx, gr, gl, gx)/sig0(xg, s, mq, Mmed, mx, gr, gl, gx) )
#         return result

#     a = 0.04
#     b = 0.66
#     #a = 0.001
#     #b = 0.100

#     INTEGRAL = integrand_xg(xg, s, mq, Mmed, mx, gr, gl, gx, sig0)

#     return INTEGRAL     



# @np.vectorize
# def HPhsig_xmin(xmin, xg, s, mq, Mmed, mx, gr, gl, gx, sig0):
#     @np.vectorize
#     def integrand_xg(xg, s, mq, Mmed, mx, gr, gl, gx, sig0):
#         q0 = xg*np.sqrt(s)/2

#         return ((1/q0) * (1 - (q0 /((np.sqrt(s)/2))) + ((q0**2) / (s/2) ) ))* (sig0(xg, s*(1-xg), mq, Mmed, mx, gr, gl, gx)/sig0(xg, s, mq, Mmed, mx, gr, gl, gx) )

#     #a = (0.060 * (2/np.sqrt(s))) ## E_min = 0.060 TeV
#     b = (1 * (2/np.sqrt(s))) ## E_max 1 TeV
#     A = xmin*np.sqrt(s)/2
#     a = xmin
#     INTEGRAL = quad(integrand_xg, a, b, args=(s, mq, Mmed, mx, gr, gl, gx, sig0), limit=intlimit, epsabs=erroabs, epsrel=errorel)[0] ## x1 integration!! - The result is tau*\integral{f(x)} (mq is set to 0 because python errors in this phase)      
    
#     hp = sig0(xg, s, mq, Mmed, mx, gr, gl, gx) * (1 + (2*alph/np.pi) * ( ((-1 + 2*np.log10( np.sqrt(s)/mq)) *  ( np.log10(A/(np.sqrt(s)/2)) + 13/12 + INTEGRAL)) - 17/36 + ((1/6)*(np.pi**2))    ))

#     return hp

# ### (1 + delta)
# @np.vectorize
# def oneplusdelta(xg_min, xgamma, s, mq, Mmed, mx, gr, gl, gx, sig0):

#     def integrand_xg(xg, s, mq, Mmed, mx, gr, gl, gx, sig0):
#         q0 = xg*np.sqrt(s)/2
#         #print('q0', q0)
#         result = ((1/q0) * (1 - (q0 /((np.sqrt(s)/2))) + ((q0**2) / (s/2) ) ))* (sig0(xg, s*(1-xg), mq, Mmed, mx, gr, gl, gx)/sig0(xg, s, mq, Mmed, mx, gr, gl, gx) )
#         return result
#     #a = 0.060 * (2/np.sqrt(s))  ## E_min = 0.060 TeV
#     #b = 0.66
#     #b =  (1 - ((2*mx**2)/s))  ## E_gamma_max
#     b = (1 * (2/np.sqrt(s)))


#     #INTEGRAL = quad(integrand_xg, xg_min, b, args=(s, mq, Mmed, mx, gr, gl, gx, sig0), limit=intlimit, epsabs=erroabs, epsrel=errorel)[0] ## x1 integration!! - The result is tau*\integral{f(x)} (mq is set to 0 because python errors in this phase)      
#     def i_argument(x):
#         result = integrand_xg(x, s, mq, Mmed, mx, gr, gl, gx, sig0)
#         return result
    
#     INTEGRAL = fortran_quad.quad16(i_argument, xg_min, b)
#     A = xg_min*np.sqrt(s)/2
#     hp = (1 + (2*alph/np.pi) * ( ((-1 + 2*np.log10( np.sqrt(s)/mq)) *  ( np.log10(A/(np.sqrt(s)/2)) + 13/12 + INTEGRAL)) - 17/36 + ((1/6)*(np.pi**2))    ))
#     #print(INTEGRAL)
#     return hp



# @np.vectorize
# def WW_hsig(xg, s, mq, Mmed, mx, gr, gl, gx, sig0):   
#     try:
#         ww = (2*alph/np.pi) * ((1+(xg - 1)**2)/xg) * np.log(s/mq) * (2/s) * sig0(xg, (1-xg)*s, mq, Mmed, mx, gr, gl, gx)
#     except ZeroDivisionError:
#         ww = 0 
#     return ww


# @np.vectorize    ##call the function without any factorization
# def no_fact(xg, s, mq, Mmed, mx, gr, gl, gx, sig0):   
    
#     return sig0(xg, s, mq, Mmed, mx, gr, gl, gx)




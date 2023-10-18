import numpy as np
import scipy as sp
import fortran_quad
import models.sigma0_xsections as sig0


######################### RELIC DENSITY CALC ###############################3333

##!! WARNING: sigma must be in TeV^-2 units, not barn ##
h = 67.8 / 100     ## constante de hubble reduzida - Planck 2018
Mplanck = 1.220910*(10**16) # Massa de Planck em TeV - PDG 2017
s0 = 2891.1 /  (5.06773071*(10**16))**3  # entropy density/Boltzmann constant, em TeV: 1 cm = 5.06773071*(10^16) TeV^-1
#n = 0
gf = 1   # número de graus de liberdade:spin, cor, etc. do candidato de ME -- 1 = escalar, 2 = fermion, 3 = vetorial
gst = 86.25 # g*, Número de partículas relativisticas no periodo do desacoplamento MAX = 106.75, DM101 = 5 GeV < T < 80 GeV, g = 86.25
rho =  1.053672*(10**(-5))*(h**2)*(10**(-3)) / (5.06773071*(10**16))**3     #usando que 1 cm = 5.06773071*(10^16) TeV^-1
GN = 1 / Mplanck**2
T0 = 2.7255 * (8.61732814974056E-05)  / 10E12  # CMB temperature in K to eV, then to TeV
gff = 86.25
planckdata1 = [0.120]   #PLANCK data for \Omega h^2 = 0.120 (2018)
planckdata2 = [[0.120, 1E1000], [0.120]] 

## NEW CALCULATION FOR RELIC DENSITY
from scipy.integrate import quad
erroabs = 1
errorel = 1
import scipy.special as sp
#from sympy import *

## Implementation of: arXiv:1912.02870v2 and Gondolo 1991


xf = 28  ## Following https://arxiv.org/pdf/1703.05703.pdf
#Tf = mx / xf
intlimit = 10

import scipy
from numpy import real
## To fix complex bugs with quadrature


def complex_quadrature(func, a, b, **kwargs):

    def real_func(eps, x):
        real_result = np.real(complex(func(eps, x)))
        return real_result 

    def imag_func(eps, x):
        imag_result = np.imag(complex(func(eps, x)))
        return imag_result

    real_integral = quad(real_func, a, b, **kwargs)
    imag_integral = quad(imag_func, a, b, **kwargs)
    result_cq = (real_integral[0] + 1j*imag_integral[0], real_integral[1:], imag_integral[1:])
    return result_cq

@np.vectorize
def complex_quadrature_fortran(func, a, b , x):

    def real_func(eps):
        real_result = np.real(complex(func(eps, x)))
        return real_result 

    def imag_func(eps):
        imag_result = np.imag(complex(func(eps, x)))
        return imag_result


    real_integral = fortran_quad.quad16(real_func, a, b)
    imag_integral = fortran_quad.quad16(imag_func, a, b)


    result_cq = (real_integral + 1j*imag_integral)
    return result_cq


#### Ressonance <sigma.v> - Gondolo 1991, eq 6.13 ####
@np.vectorize
def sigmav_res(T, s, mq, Mmed, mx, gr, gl, gx, dmname, Nfermions):
    x = mx/T
    # spin configuration
    dict_spin = {'Scalar':0, 'Fermion': 1/2, 'Vector':1}
    spin_j = dict_spin[dmname] ## Spin of colliding particles: 0 - Scalar, 1/2 - Fermion, 1 - Vector
    #spin_j = 1/2 ## Spin of colliding particles: 0 - Scalar, 1/2 - Fermion, 1 - Vector
    spin_s = 1 ## Spin of mediator / ressonance
    W = (2*spin_j + 1)/((2*spin_s  + 1)**2)

    TotalDecay = sig0.GM(dmname).GMdecay(s, mq, Mmed, mx, gr, gl, gx) + sig0.GM('SM').GMdecay(s, mq, Mmed, mx, gr, gl, gx, Nfermions)

    epsR = ((Mmed**2) - (4*mx**2))/(4*mx**2)
    gamma_r = (Mmed * TotalDecay) / (4*mx**2)

    Bi = sig0.GM(dmname).GMdecay(s, mq, Mmed, mx, gr, gl, gx) / (TotalDecay)
    #br = GM(dmname).GMdecay(s, mq, Mmed, mx, gr, gl, gx) * (1 - GM(dmname).GMdecay(s, mq, Mmed, mx, gr, gl, gx) )
    Br = Bi * (1 - Bi)


    @np.vectorize
    def Flzrx(T):

        def Flzrx_U(u, T):
    
            x = mx/T

            eps = (- np.log(u)) / x
            epsR = ((Mmed**2) - (4*mx**2))/(4*mx**2)
            gamma_r = (Mmed * TotalDecay) / (4*mx**2)
            zr = epsR + (1j)*gamma_r

            result = ((np.sqrt(1 + eps))/(1 + 2*eps)) * (1/(zr - eps)) * (1/x)
            return  result

        result_flzrx = np.real(complex((1j/(np.pi)) * complex_quadrature_fortran(Flzrx_U, 0, 1, T)))
        return result_flzrx

    prefact1 = (16*(np.pi)/(mx**2)) * W
    prefact2 = (x**(3/2)) * ((np.pi) ** (1/2)) * gamma_r * Br

    result = prefact1 * prefact2 * Flzrx(T)
    return result

@np.vectorize
def Ohm_numerical(s, mq, Mmed, mx, gr, gl, gx, dmname, Nfermions = 6):
    if mx > Mmed/2:
        return np.nan
    else:
        
        def i_argument(x):
            result = sigmav_res(x, s, mq, Mmed, mx, gr, gl, gx, dmname, Nfermions = 6)
            return result
        

        integral = fortran_quad.quad16(i_argument, T0, mx/xf)
        #integral = quad(sigmav_res, T0, mx/xf, args=( s, mq, Mmed, mx, gr, gl, gx, dmname), limit=intlimit, epsabs=erroabs, epsrel=errorel)[0]
        result = 8.76 * (1E-11)  * (1/((np.sqrt(gst)*integral)/mx)) * (10**6) ## conversion from GeV^-2 to TeV^-2

        return result


@np.vectorize
def sigmav_gondolo(T, s, mq, Mmed, mx, gr, gl, gx, dmname, Bi, l):
    x = mx/T
    # spin configuration
    dict_spin = {'Scalar':0, 'Fermion': 1/2, 'Vector':1}
    spin_j = dict_spin[dmname] ## Spin of colliding particles: 0 - Scalar, 1/2 - Fermion, 1 - Vector
    
    spin_s = 1 ## Spin of mediator / ressonance
    W = (2*spin_j + 1)/((2*spin_s  + 1)**2)

    TotalDecay = sig0.GM(dmname).GMdecay(s, mq, Mmed, mx, gr, gl, gx) + sig0.GM('SM').GMdecay(s, mq, Mmed, mx, gr, gl, gx)
    
    
    #Td = 0.02 * Mmed
    

    #TotalDecay = Td

    epsR = ((Mmed**2) - (4*mx**2))/(4*mx**2)
    gamma_r = (Mmed * TotalDecay) / (4*mx**2)


    #Bi = GM(dmname).GMdecay(s, mq, Mmed, mx, gr, gl, gx) / (TotalDecay)
    #Bi = 0.01
    #br = GM(dmname).GMdecay(s, mq, Mmed, mx, gr, gl, gx) * (1 - GM(dmname).GMdecay(s, mq, Mmed, mx, gr, gl, gx) )
    Br = Bi * (1 - Bi)


    @np.vectorize
    def Flzrx(T, l):

        def Flzrx_U(u, T):
    
            x = mx/T

            eps = (- np.log(u)) / x
            epsR = ((Mmed**2) - (4*mx**2))/(4*mx**2)
            gamma_r = (Mmed * TotalDecay) / (4*mx**2)
            zr = epsR + (1j)*gamma_r
            
            
            
            num = (eps**(l + (1/2))) 
            denom = zr - eps  
                                        
            #result = ((np.sqrt(1 + eps))/(1 + 2*eps)) * (1/(zr - eps)) * (1/x)
            result_gon = (num / denom) * (1/x)
            return  result_gon

        #result_flzrx = np.real(complex((1j/(np.pi)) * complex_quadrature_fortran(Flzrx_U, 0, 1, T)))
        erroabs = 0.00001
        errorel = 0.00001
        intlimit = 100
        result_flzrx = np.real(complex((1j/(np.pi)) * complex_quadrature(Flzrx_U, 0, 1, args=(T,), limit=intlimit, epsabs=erroabs, epsrel=errorel)[0]))
        
        return result_flzrx

    prefact1 = (16*(np.pi)/(mx**2)) * W
    #prefact2 = (x**(3/2)) * ((np.pi) ** (1/2)) * gamma_r * Br

    #result = prefact1 * prefact2 * Flzrx(T)
    
    prefact3 = 2*(x**(3/2)) * ((np.pi) ** (-1/2)) # * Br


    result = prefact3  * Flzrx(T, l)
    
    return result














### not used ###

# def oldOHM_relic(sigma, mx, n):    ## cálculo da relic density em cada ponto 
#     xfo = np.log(0.038*(n+1)*(gf/(gst**(1/2)))*Mplanck*mx*sigma) - (n+0.5)*np.log(np.log(0.038*(n+1)*(gf/(gst**(0.5)))*Mplanck*mx*sigma))
#     OHM_dm = ((mx*s0)/rho) * ( 3.79*(n+1)*(xfo**(n+1)) ) / ((gst/gst**(1/2))*Mplanck*mx*sigma)
#     OHM_dm = OHM_dm*(h**2)
#     return OHM_dm
# ##!! WARNING: sigma must be in TeV^-2 units, not barn ##
# ################################################################################
################################################################################


# @np.vectorize
# def sigmav_x(sig0, x, xg, s, mq, Mmed, mx, gr, gl, gx):

    
#     def int0(s, mq, Mmed, mx, gr, gl, gx):
#         prefac = (2*np.pi**2 * (mx/x)) / ( (4 * np.pi * mx**2 * (mx/x) * (sp.kn(2, x))) **2 )
    
#         newprefac = (1 / (8 * (mx**4) * (mx/x) * ((sp.kn(2, x))** 2) )) 
    
#         return newprefac * (sig0(xg, s, mq, Mmed, mx, gr, gl, gx) * np.sqrt(s)* (s - 4*mx**2) * sp.kn(1, np.sqrt(s)*(x/mx)))

    
#     factcmframe = ( (1/2)*(1 + ( (sp.kn(1, x)**2)/(sp.kn (2,  x)**2) )) ) 

#     ## Integrates int0 over s (first variable), from 4mx² to inf 
#     ## sigmav0 =  intsimp(int0, 4*mx**2, inff, Npoints, mq, Mmed, mx, gr, gl, gx)
#     sigmav0 =  quad(int0, 4*mx**2, np.inf, args=(mq, Mmed, mx, gr, gl, gx), limit=intlimit, epsabs=erroabs, epsrel=errorel)[0]

#     return factcmframe * sigmav0
    


# @np.vectorize
# def Za1(x, sig0, xg, s, mq, Mmed, mx, gr, gl, gx):
#     result = np.sqrt((np.pi/45)*gst) * (mx/x**2) * Mplanck * sigmav_x(sig0, x, xg, s, mq, Mmed, mx, gr, gl, gx)
#     if np.isnan(result):
#         return 0
#     else:
#         return result


# # ## Functions to calculate xf analytically
# # @np.vectorize
# # def Yeq(x):
# #     return (45/4*np.pi**4) *  ((x**2 *sp.kn(2, x)) / heff)

# # @np.vectorize
# # def Yhat(x):
# #     return np.exp(x)*Yeq(x)

# # @np.vectorize
# # def Yeq_prime(x): # From mathematica
# #     cte = (45/4*np.pi**4)/heff
# #     result = np.exp(x) * cte * (x**2) * sp.kn(2,x) + (1/2) * np.exp(x) * cte * (x**2) * (-sp.kn(1,x) -sp.kn(3,x))
# #     return result

# # @np.vectorize
# # def xF(x, xg, s, mq, Mmed, mx, gr, gl, gx):
# #     logint = (3/2)* ( (Za(x, xg, s, mq, Mmed, mx, gr, gl, gx) * (Yeq(x)**2)) / (Yeq(x) - Yeq_prime(x)) )

# #     return np.log10(logint)

# @np.vectorize
# def Ytoday1(sig0, xg, s, mq, Mmed, mx, gr, gl, gx):
    
#     YEQxf = (45/4*np.pi**4) *  ((xf**2 *sp.kn(2, xf)) / heff)
    
#     Yf = (1 + deltaf) * (YEQxf)
    
#     Af = quad(Za1, xf, inff, args=(sig0, xg, s, mq, Mmed, mx, gr, gl, gx), limit=intlimit, epsabs=erroabs, epsrel=errorel)[0]
    
#     Ytoday =  Yf / (1 + (Yf * Af))
    
#     return Ytoday


# @np.vectorize
# def OHM_relic(sig0, xg , s, mq, Mmed, mx, gr, gl, gx):
#     #Fixed parameters
    
#     Npoints = 40
#     xf = 28
#     Tf = mx / xf
#     heff = 1
#     deltaf = 1
#     inff = 100
#     ohmprefac = (mx * s0 * h**2) / rho 
#     newOHM_dm = ohmprefac  * Ytoday1(sig0, xg, s, mq, Mmed, mx, gr, gl, gx)


#     ### Correct for the naive calculation from Gondolo (1991)
#     correction = (4*mx**2)/(xf*(np.pi**(3/2))* Mmed * GMF(xg, s, mq, Mmed, mx, gr, gl, gx)  )
#     Ohm_correct =  correction * newOHM_dm

#     return Ohm_correct





        

# @np.vectorize
# def Za(x, xg, s, mq, Mmed, mx, gr, gl, gx, dmname):
#     result = np.sqrt((np.pi/45)*gst) * (mx/x**2) * Mplanck * sigmav_x(x, xg, s, mq, Mmed, mx, gr, gl, gx, dmname)
#     if np.isnan(result):
#         return 0
#     else:
#         return result

# @np.vectorize
# def Ytoday(xg, s, mq, Mmed, mx, gr, gl, gx, dmname):
    
#     YEQxf = (45/4*np.pi**4) *  ((xf**2 *sp.kn(2, xf)) / heff)
    
#     Yf = (1 + deltaf) * (YEQxf)
    
#     Af = quad(Za, xf, inff, args=(xg, s, mq, Mmed, mx, gr, gl, gx, dmname), limit=intlimit, epsabs=erroabs, epsrel=errorel)[0]
    
#     Ytoday =  Yf / (1 + (Yf * Af))
    
#     return Ytoday


# @np.vectorize
# def newOHM_relic(xg, s, mq, Mmed, mx, gr, gl, gx, dmname):
#     #Fixed parameters
    
#     Npoints = 40
#     xf = 28
#     Tf = mx / xf
#     heff = 1
#     deltaf = 1
#     inff = 100
#     ohmprefac = (mx * s0 * h**2) / rho 
#     newOHM_dm = ohmprefac  * Ytoday(xg, s, mq, Mmed, mx, gr, gl, gx, dmname)

#     print(newOHM_dm)
#     return newOHM_dm

########################################################
######################################################
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
def sigmav_res(T, s, mq, Mmed, mx, gr, gl, gx, dmname, Nf):
    x = mx/T
    # spin configuration
    dict_spin = {'Scalar':0, 'Fermion': 1/2, 'Vector':1}
    spin_j = dict_spin[dmname] ## Spin of colliding particles: 0 - Scalar, 1/2 - Fermion, 1 - Vector
    #spin_j = 1/2 ## Spin of colliding particles: 0 - Scalar, 1/2 - Fermion, 1 - Vector
    spin_s = 1 ## Spin of mediator / ressonance
    W = (2*spin_j + 1)/((2*spin_s  + 1)**2)

    TotalDecay = sig0.GM(dmname).GMdecay(s, mq, Mmed, mx, gr, gl, gx, Nf) + sig0.GM('SM').GMdecay(s, mq, Mmed, mx, gr, gl, gx, Nf)

    epsR = ((Mmed**2) - (4*mx**2))/(4*mx**2)
    gamma_r = (Mmed * TotalDecay) / (4*mx**2)

    Bi = sig0.GM(dmname).GMdecay(s, mq, Mmed, mx, gr, gl, gx, Nf) / (TotalDecay)
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
def Ohm_numerical(s, mq, Mmed, mx, gr, gl, gx, dmname, Nf):
    if mx > Mmed/2:
        return np.nan
    else:
        
        def i_argument(x):
            result = sigmav_res(x, s, mq, Mmed, mx, gr, gl, gx, dmname, Nf)
            return result
        

        integral = fortran_quad.quad16(i_argument, T0, mx/xf)
        #integral = quad(sigmav_res, T0, mx/xf, args=( s, mq, Mmed, mx, gr, gl, gx, dmname), limit=intlimit, epsabs=erroabs, epsrel=errorel)[0]
        result = 8.76 * (1E-11)  * (1/((np.sqrt(gst)*integral)/mx)) * (10**6) ## conversion from GeV^-2 to TeV^-2

        return result


@np.vectorize
def sigmav_gondolo(T, s, mq, Mmed, mx, gr, gl, gx, dmname, Bi, l, Nf):
    x = mx/T
    # spin configuration
    dict_spin = {'Scalar':0, 'Fermion': 1/2, 'Vector':1}
    spin_j = dict_spin[dmname] ## Spin of colliding particles: 0 - Scalar, 1/2 - Fermion, 1 - Vector
    
    spin_s = 1 ## Spin of mediator / ressonance
    W = (2*spin_j + 1)/((2*spin_s  + 1)**2)

    TotalDecay = sig0.GM(dmname).GMdecay(s, mq, Mmed, mx, gr, gl, gx, Nf) + sig0.GM('SM').GMdecay(s, mq, Mmed, mx, gr, gl, gx, Nf)
    
    
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

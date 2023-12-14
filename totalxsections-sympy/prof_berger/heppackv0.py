#
from sympy.combinatorics.permutations import Permutation
from sympy import *
from IPython.display import display
#from torch import JitType

init_printing()
#
#
print("Reading heppackv0.py (March 2023)")
print()
#
Eins = Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
#
g0 = Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]])
g1 = Matrix([[0, 0, 0, 1], [0, 0, 1, 0], [0, -1, 0, 0], [-1, 0, 0, 0]])
g2 = Matrix([[0, 0, 0, -I], [0, 0, I, 0], [0, I, 0, 0], [-I, 0, 0, 0]])
g3 = Matrix([[0, 0, 1, 0], [0, 0, 0, -1], [-1, 0, 0, 0], [0, 1, 0, 0]])
# g5=I*g0*g1*g2*g3
g5 = Matrix([[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]])
one = Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
projpl = (one + g5) / 2
projm = (one - g5) / 2

projR = (one + g5) / 2
projL = (one - g5) / 2
#
# Pauli Matrices (compare book chapter 2.2 or 2.8)
#
sig1 = Matrix([[0, 1], [1, 0]])
sig2 = Matrix([[0, -I], [I, 0]])
sig3 = Matrix([[1, 0], [0, -1]])


#
def chi1(theta, phi):
    elm1 = cos(theta / 2)
    elm2 = exp(I * phi) * sin(theta / 2)
    result = Matrix([[elm1], [elm2]])
    return result


def chi2(theta, phi):
    elm1 = -exp(-I * phi) * sin(theta / 2)
    elm2 = cos(theta / 2)
    result = Matrix([[elm1], [elm2]])
    return result


#
# Helicity-spinors (compare book chapter 3.1)
#
def u_min(Ef, mf, theta, phi):
    elm1 = -exp(-I * phi) * sin(theta / 2)
    elm2 = cos(theta / 2)
    fac1 = sqrt(Ef - mf) / sqrt(Ef + mf)
    fac2 = sqrt(Ef + mf)
    elm3 = -fac1 * elm1
    elm4 = -fac1 * elm2
    result = fac2 * Matrix([[elm1], [elm2], [elm3], [elm4]])
    return result


#
def u_plus(Ef, mf, theta, phi):
    elm2 = exp(I * phi) * sin(theta / 2)
    elm1 = cos(theta / 2)
    fac1 = sqrt(Ef - mf) / sqrt(Ef + mf)
    fac2 = sqrt(Ef + mf)
    elm3 = fac1 * elm1
    elm4 = fac1 * elm2
    result = fac2 * Matrix([[elm1], [elm2], [elm3], [elm4]])
    return result


#
def v_min(Ef, mf, theta, phi):
    elm4 = exp(I * phi) * sin(theta / 2)
    elm3 = cos(theta / 2)
    fac1 = sqrt(Ef - mf) / sqrt(Ef + mf)
    fac2 = sqrt(Ef + mf)
    elm1 = fac1 * elm3
    elm2 = fac1 * elm4
    result = fac2 * Matrix([[elm1], [elm2], [elm3], [elm4]])
    return result


#
def v_plus(Ef, mf, theta, phi):
    elm3 = -exp(-I * phi) * sin(theta / 2)
    elm4 = cos(theta / 2)
    fac1 = sqrt(Ef - mf) / sqrt(Ef + mf)
    fac2 = sqrt(Ef + mf)
    elm1 = -fac1 * elm3
    elm2 = -fac1 * elm4
    result = fac2 * Matrix([[elm1], [elm2], [elm3], [elm4]])
    return result


#
def u_minbar(Ef, mf, theta, phi):
    h1 = u_min(Ef, mf, theta, phi).T * g0
    elm1 = h1[0] * exp(2 * I * phi)
    elm2 = h1[1]
    # fac1=sqrt(Ef-mf)/sqrt(Ef+mf)
    # fac2=sqrt(Ef+mf)
    elm3 = h1[2] * exp(2 * I * phi)
    elm4 = h1[3]
    result = Matrix([[elm1, elm2, elm3, elm4]])
    return result


#
def u_plusbar(Ef, mf, theta, phi):
    h1 = u_plus(Ef, mf, theta, phi).T * g0
    elm2 = h1[1] * exp(-2 * I * phi)
    elm1 = h1[0]
    # fac1=sqrt(Ef-mf)/sqrt(Ef+mf)
    # fac2=sqrt(Ef+mf)
    elm4 = h1[3] * exp(-2 * I * phi)
    elm3 = h1[2]
    result = Matrix([[elm1, elm2, elm3, elm4]])
    return result


#
def v_minbar(Ef, mf, theta, phi):
    h1 = v_min(Ef, mf, theta, phi).T * g0
    elm2 = h1[1] * exp(-2 * I * phi)
    elm1 = h1[0]
    # fac1=sqrt(Ef-mf)/sqrt(Ef+mf)
    # fac2=sqrt(Ef+mf)
    elm4 = h1[3] * exp(-2 * I * phi)
    elm3 = h1[2]
    result = Matrix([[elm1, elm2, elm3, elm4]])
    return result


#
def v_plusbar(Ef, mf, theta, phi):
    h1 = v_plus(Ef, mf, theta, phi).T * g0
    elm1 = h1[0] * exp(2 * I * phi)
    elm2 = h1[1]
    # fac1=sqrt(Ef-mf)/sqrt(Ef+mf)
    # fac2=sqrt(Ef+mf)
    elm3 = h1[2] * exp(2 * I * phi)
    elm4 = h1[3]
    result = Matrix([[elm1, elm2, elm3, elm4]])
    return result


#
def u(pp, ha):
    if ha >= 0:
        result = u_plus(pp[0], pp[1], pp[2], pp[3])
    elif ha < 0:
        result = u_min(pp[0], pp[1], pp[2], pp[3])
    return result


#
def v(pp, ha):
    if ha >= 0:
        result = v_plus(pp[0], pp[1], pp[2], pp[3])
    elif ha < 0:
        result = v_min(pp[0], pp[1], pp[2], pp[3])
    return result


#
def ubar(pp, ha):
    if ha >= 0:
        result = u_plusbar(pp[0], pp[1], pp[2], pp[3])
    elif ha < 0:
        result = u_minbar(pp[0], pp[1], pp[2], pp[3])
    return result


#
def vbar(pp, ha):
    if ha >= 0:
        result = v_plusbar(pp[0], pp[1], pp[2], pp[3])
    elif ha < 0:
        result = v_minbar(pp[0], pp[1], pp[2], pp[3])
    return result


#
# Polarization vectors for photons and W bosons (compare book chapter 2.4
# and chapter 3.2)
#
def pol_plus(theta, phi):
    elm1 = 0
    h1 = cos(theta / 2) ** 2 - sin(theta / 2) ** 2
    h2 = 2 * sin(theta / 2) * cos(theta / 2)
    elm2 = sqrt(2) * cos(phi) * h1 / 2 - I * sqrt(2) * sin(phi) / 2
    elm3 = sqrt(2) * sin(phi) * h1 / 2 + I * sqrt(2) * cos(phi) / 2
    elm4 = -sqrt(2) * h2 / 2
    result = [elm1, elm2, elm3, elm4]
    return Matrix(result)


#
#
def pol_min(theta, phi):
    elm1 = 0
    h1 = cos(theta / 2) ** 2 - sin(theta / 2) ** 2
    h2 = 2 * sin(theta / 2) * cos(theta / 2)
    elm2 = sqrt(2) * cos(phi) * h1 / 2 + I * sqrt(2) * sin(phi) / 2
    elm3 = sqrt(2) * sin(phi) * h1 / 2 - I * sqrt(2) * cos(phi) / 2
    elm4 = -sqrt(2) * h2 / 2
    result = [elm1, elm2, elm3, elm4]
    return Matrix(result)


#
#
def pol_0(e0, m0, theta, phi):
    elm1 = sqrt(e0**2 - m0**2) / m0
    elm2 = e0 * sin(theta) * cos(phi) / m0
    elm3 = e0 * sin(theta) * sin(phi) / m0
    elm4 = e0 * cos(theta) / m0
    result = [elm1, elm2, elm3, elm4]
    return Matrix(result)


#
def pol(pp, ha):
    e0 = pp[0]
    m0 = pp[1]
    theta = pp[2]
    phi = pp[3]
    if ha > 0:
        result = pol_plus(theta, phi)
    elif ha < 0:
        result = pol_min(theta, phi)
    else:
        result = pol_0(e0, m0, theta, phi)

    return result


#
def polbar(pp, ha):
    e0 = pp[0]
    m0 = pp[1]
    theta = pp[2]
    phi = pp[3]
    if ha > 0:
        result = pol_min(theta, phi)
        display(f'+ pol {result}')
    elif ha < 0:
        result = pol_plus(theta, phi)
        display(f'- pol {result}')
    else:
        result = pol_0(e0, m0, theta, phi)
        display(f'0 pol {result}')
    return transpose(result) ## Modified 2023-12-13, add transpose conjugation


#
# useful routines for calculations:
#
# fourvec calculates E, p_x,p_y,p_z from E, m, theta, phi
#
def fourvec(pp):
    elm1 = pp[0]
    elm2 = pp[1]
    elm3 = pp[2]
    elm4 = pp[3]

    h0 = elm3 / 2
    h1 = 2 * sin(h0) * cos(h0)
    h2 = (cos(h0) ** 2) - (sin(h0) ** 2)
    p = sqrt(elm1**2 - elm2**2)
    #result = [elm1, p * h1 * cos(elm4), p * h1 * sin(elm4), p * h2] -- Marcio: Adiconei matrix aqui
    result = Matrix([elm1, p * h1 * cos(elm4), p * h1 * sin(elm4), p * h2])
    return result


#
#
# dotprod4 evaluates the calar product of 2 fourvectors
#
def dotprod4(jj1, jj2):
    h1 = jj1[0] * jj2[0] - jj1[1] * jj2[1] - jj1[2] * jj2[2] - jj1[3] * jj2[3]
    result = simplify(h1)
    return result


#
# dag evaluates the dagger matrix as defined in chapter 3.1
# #MARCIO NOTE: This is in fact what we call the "slash"
def dag(pp):
    elm1 = pp[0]
    elm2 = pp[1]
    elm3 = pp[2]
    elm4 = pp[3]
    result = elm1 * g0 - elm2 * g1 - elm3 * g2 - elm4 * g3
    return result


#
# currents gamma_mu coupling (QED, compare book chapter 3.2)
#
def ubv(p1, h1, p2, h2):
    j0 = ubar(p1, h1) * g0 * v(p2, h2)
    j1 = ubar(p1, h1) * g1 * v(p2, h2)
    j2 = ubar(p1, h1) * g2 * v(p2, h2)
    j3 = ubar(p1, h1) * g3 * v(p2, h2)
    result = [simplify(j0[0]), simplify(j1[0]), simplify(j2[0]), simplify(j3[0])]
    return result


#
def ubu(p1, h1, p2, h2):
    j0 = ubar(p1, h1) * g0 * u(p2, h2)
    j1 = ubar(p1, h1) * g1 * u(p2, h2)
    j2 = ubar(p1, h1) * g2 * u(p2, h2)
    j3 = ubar(p1, h1) * g3 * u(p2, h2)
    result = [simplify(j0[0]), simplify(j1[0]), simplify(j2[0]), simplify(j3[0])]
    return result


#
def vbu(p1, h1, p2, h2):
    j0 = vbar(p1, h1) * g0 * u(p2, h2)
    j1 = vbar(p1, h1) * g1 * u(p2, h2)
    j2 = vbar(p1, h1) * g2 * u(p2, h2)
    j3 = vbar(p1, h1) * g3 * u(p2, h2)
    result = [simplify(j0[0]), simplify(j1[0]), simplify(j2[0]), simplify(j3[0])]
    return result


#
def vbv(p1, h1, p2, h2):
    j0 = vbar(p1, h1) * g0 * v(p2, h2)
    j1 = vbar(p1, h1) * g1 * v(p2, h2)
    j2 = vbar(p1, h1) * g2 * v(p2, h2)
    j3 = vbar(p1, h1) * g3 * v(p2, h2)
    result = [simplify(j0[0]), simplify(j1[0]), simplify(j2[0]), simplify(j3[0])]
    return result


#
# charged current weak interaction, (1-g5) coupling (compare book chapter 6.1)
#
#
def ubvw(p1, h1, p2, h2):
    j0 = ubar(p1, h1) * g0 * projm * v(p2, h2)
    j1 = ubar(p1, h1) * g1 * projm * v(p2, h2)
    j2 = ubar(p1, h1) * g2 * projm * v(p2, h2)
    j3 = ubar(p1, h1) * g3 * projm * v(p2, h2)
    result = [simplify(j0[0]), simplify(j1[0]), simplify(j2[0]), simplify(j3[0])]
    return result


#
#
def ubuw(p1, h1, p2, h2):
    j0 = ubar(p1, h1) * g0 * projm * u(p2, h2)
    j1 = ubar(p1, h1) * g1 * projm * u(p2, h2)
    j2 = ubar(p1, h1) * g2 * projm * u(p2, h2)
    j3 = ubar(p1, h1) * g3 * projm * u(p2, h2)
    result = [simplify(j0[0]), simplify(j1[0]), simplify(j2[0]), simplify(j3[0])]
    return result


#
#
def vbuw(p1, h1, p2, h2):
    j0 = vbar(p1, h1) * g0 * projm * u(p2, h2)
    j1 = vbar(p1, h1) * g1 * projm * u(p2, h2)
    j2 = vbar(p1, h1) * g2 * projm * u(p2, h2)
    j3 = vbar(p1, h1) * g3 * projm * u(p2, h2)
    result = [simplify(j0[0]), simplify(j1[0]), simplify(j2[0]), simplify(j3[0])]
    return result


#
def vbvw(p1, h1, p2, h2):
    j0 = vbar(p1, h1) * g0 * projm * v(p2, h2)
    j1 = vbar(p1, h1) * g1 * projm * v(p2, h2)
    j2 = vbar(p1, h1) * g2 * projm * v(p2, h2)
    j3 = vbar(p1, h1) * g3 * projm * v(p2, h2)
    result = [simplify(j0[0]), simplify(j1[0]), simplify(j2[0]), simplify(j3[0])]
    return result


#
# currents cv-ca*g5 coupling (compare book chapter 6.4)
#
def ubvva(p1, h1, p2, h2, cv, ca):
    gv = cv / 2
    ga = ca / 2
    j0 = ubar(p1, h1) * g0 * (gv * one - ga * g5) * v(p2, h2)
    j1 = ubar(p1, h1) * g1 * (gv * one - ga * g5) * v(p2, h2)
    j2 = ubar(p1, h1) * g2 * (gv * one - ga * g5) * v(p2, h2)
    j3 = ubar(p1, h1) * g3 * (gv * one - ga * g5) * v(p2, h2)
    result = [simplify(j0[0]), simplify(j1[0]), simplify(j2[0]), simplify(j3[0])]
    return result


#
#
def ubuva(p1, h1, p2, h2, cv, ca):
    gv = cv / 2
    ga = ca / 2
    j0 = ubar(p1, h1) * g0 * (gv * one - ga * g5) * u(p2, h2)
    j1 = ubar(p1, h1) * g1 * (gv * one - ga * g5) * u(p2, h2)
    j2 = ubar(p1, h1) * g2 * (gv * one - ga * g5) * u(p2, h2)
    j3 = ubar(p1, h1) * g3 * (gv * one - ga * g5) * u(p2, h2)
    result = [simplify(j0[0]), simplify(j1[0]), simplify(j2[0]), simplify(j3[0])]
    return result


##
def vbuva(p1, h1, p2, h2, cv, ca):
    gv = cv / 2
    ga = ca / 2
    j0 = vbar(p1, h1) * g0 * (gv * one - ga * g5) * u(p2, h2)
    j1 = vbar(p1, h1) * g1 * (gv * one - ga * g5) * u(p2, h2)
    j2 = vbar(p1, h1) * g2 * (gv * one - ga * g5) * u(p2, h2)
    j3 = vbar(p1, h1) * g3 * (gv * one - ga * g5) * u(p2, h2)
    result = [simplify(j0[0]), simplify(j1[0]), simplify(j2[0]), simplify(j3[0])]
    return result


#
#
def vbvva(p1, h1, p2, h2, cv, ca):
    gv = cv / 2
    ga = ca / 2
    j0 = vbar(p1, h1) * g0 * (gv * one - ga * g5) * v(p2, h2)
    j1 = vbar(p1, h1) * g1 * (gv * one - ga * g5) * v(p2, h2)
    j2 = vbar(p1, h1) * g2 * (gv * one - ga * g5) * v(p2, h2)
    j3 = vbar(p1, h1) * g3 * (gv * one - ga * g5) * v(p2, h2)
    result = [simplify(j0[0]), simplify(j1[0]), simplify(j2[0]), simplify(j3[0])]
    return result


#
#


#
# Pauli currents as needed in book chapter 5.2
#
def ubuP(p1, h1, p2, h2):
    M = p1[1]
    p14 = fourvec(p1)
    p24 = fourvec(p2)
    # qmu with lower index
    q0 = p14[0] - p24[0]
    q1 = -p14[1] + p24[1]
    q2 = -p14[2] + p24[2]
    q3 = -p14[3] + p24[3]
    # sigmamunu with upper index like in Bjorken Drell
    # but without factor i/2
    jF0 = (g0 * g1 - g1 * g0) * q1 + (g0 * g2 - g2 * g0) * q2 + (g0 * g3 - g3 * g0) * q3
    jF1 = (g1 * g0 - g0 * g1) * q0 + (g1 * g2 - g2 * g1) * q2 + (g1 * g3 - g3 * g1) * q3
    jF2 = (g2 * g0 - g0 * g2) * q0 + (g2 * g1 - g1 * g2) * q1 + (g2 * g3 - g3 * g2) * q3
    jF3 = (g3 * g0 - g0 * g3) * q0 + (g3 * g1 - g1 * g3) * q1 + (g3 * g2 - g2 * g3) * q2
    j0 = ubar(p1, h1) * jF0 / 4 / M * u(p2, h2)
    j1 = ubar(p1, h1) * jF1 / 4 / M * u(p2, h2)
    j2 = ubar(p1, h1) * jF2 / 4 / M * u(p2, h2)
    j3 = ubar(p1, h1) * jF3 / 4 / M * u(p2, h2)
    # include factor i/2
    result = [-j0[0], -j1[0], -j2[0], -j3[0]]
    return result


#
#
#
# s channel and u channel amplitudes for compton scattering
#
def compts(k1, hk1, p1, hp1, k2, hk2, p2, hp2):
    ke = fourvec(k1)
    pe = fourvec(p1)
    mm = p1[1]
    mk = k1[1]
    nenner = 2 * dotprod4(ke, pe) + mk**2
    #        nenner=s0+mk^2
    epsbar = polbar(k2, hk2)
    eps = pol(k1, hk1)
    kern = dag(epsbar) * (dag(ke) + dag(pe) + mm * one) * dag(eps)
    h1 = simplify(ubar(p2, hp2) * kern * u(p1, hp1))
    result = simplify(h1[0, 0] / nenner)
    return result


#
def comptu(k1, hk1, p1, hp1, k2, hk2, p2, hp2):
    ka = fourvec(k2)
    pe = fourvec(p1)
    mm = p1[1]
    mk = k2[1]
    nenner = -2 * dotprod4(ka, pe) + mk**2
    # nenner:=u0+mk^2:
    epsbar = polbar(k2, hk2)
    eps = pol(k1, hk1)
    kern = dag(eps) * (dag(pe) - dag(ka) + mm * one) * dag(epsbar)
    h1 = simplify(ubar(p2, hp2) * kern * u(p1, hp1))
    result = simplify(h1[0, 0] / nenner)
    return result


#
def compt(k1, hk1, p1, hp1, k2, hk2, p2, hp2):
    t1 = compts(k1, hk1, p1, hp1, k2, hk2, p2, hp2)
    t2 = comptu(k1, hk1, p1, hp1, k2, hk2, p2, hp2)
    result = simplify(t1 + t2)
    return result


#
#
# Sometimes one wants the amplitudes without the propagator term
#


def Ncompts(k1, hk1, p1, hp1, k2, hk2, p2, hp2):
    ke = fourvec(k1)
    pe = fourvec(p1)
    mm = p1[1]
    mk = k1[1]
    epsbar = polbar(k2, hk2)
    eps = pol(k1, hk1)
    kern = dag(epsbar) * (dag(ke) + dag(pe) + mm * one) * dag(eps)
    h1 = simplify(ubar(p2, hp2) * kern * u(p1, hp1))
    result = simplify(h1[0, 0])
    return result


#
def Ncomptu(k1, hk1, p1, hp1, k2, hk2, p2, hp2):
    ka = fourvec(k2)
    pe = fourvec(p1)
    mm = p1[1]
    mk = k1[1]
    epsbar = polbar(k2, hk2)
    eps = pol(k1, hk1)
    kern = dag(eps) * (dag(pe) - dag(ka) + mm * one) * dag(epsbar)
    h1 = simplify(ubar(p2, hp2) * kern * u(p1, hp1))
    result = simplify(h1[0, 0])
    return result


#
#
# Now f+fbar-->gamma gamma
# p1+p2=k1+k2
#
#
def gamgamt(p1, hp1, p2, hp2, k1, hk1, k2, hk2):
    ka = fourvec(k1)
    pe = fourvec(p1)
    mm = p1[1]
    M = k1[1]
    nenner = -2 * dotprod4(ka, pe) + M**2
    epsbar2 = polbar(k2, hk2)
    epsbar1 = polbar(k1, hk1)
    kern = dag(epsbar2) * (dag(ka) - dag(pe) + mm * one) * dag(epsbar1)
    h1 = simplify(vbar(p2, hp2) * kern * u(p1, hp1))
    result = simplify(h1[0, 0] / nenner)
    return result


#
#
def gamgamu(p1, hp1, p2, hp2, k1, hk1, k2, hk2):
    ka = fourvec(k2)
    pe = fourvec(p1)
    mm = p1[1]
    M = k2[1]
    nenner = -2 * dotprod4(ka, pe) + M**2
    epsbar2 = polbar(k2, hk2)
    epsbar1 = polbar(k1, hk1)
    kern = dag(epsbar1) * (dag(ka) - dag(pe) + mm * one) * dag(epsbar2)
    h1 = simplify(vbar(p2, hp2) * kern * u(p1, hp1))
    result = simplify(h1[0, 0] / nenner)
    return result


#
#
def gamgam(p1, hp1, p2, hp2, k1, hk1, k2, hk2):
    t1 = gamgamt(p1, hp1, p2, hp2, k1, hk1, k2, hk2)
    t2 = gamgamu(p1, hp1, p2, hp2, k1, hk1, k2, hk2)
    result = simplify(t1 + t2)
    return result


#
# Amplitudes without denominator
#
#
def Ngamgamt(p1, hp1, p2, hp2, k1, hk1, k2, hk2):
    ka = fourvec(k1)
    pe = fourvec(p1)
    mm = p1[1]
    epsbar2 = polbar(k2, hk2)
    epsbar1 = polbar(k1, hk1)
    kern = dag(epsbar2) * (dag(ka) - dag(pe) + mm * one) * dag(epsbar1)
    h1 = simplify(vbar(p2, hp2) * kern * u(p1, hp1))
    result = simplify(h1[0, 0])
    return result


#
#
def Ngamgamu(p1, hp1, p2, hp2, k1, hk1, k2, hk2):
    ka = fourvec(k2)
    pe = fourvec(p1)
    mm = p1[1]
    epsbar2 = polbar(k2, hk2)
    epsbar1 = polbar(k1, hk1)
    kern = dag(epsbar1) * (dag(ka) - dag(pe) + mm * one) * dag(epsbar2)
    h1 = simplify(vbar(p2, hp2) * kern * u(p1, hp1))
    result = simplify(h1[0, 0])
    return result


##
#
# The routines V3g... and V4g... are needed for calculating the cross section
# of elastic gluon gluon scattering, gg to gg as treated in chapter 4.2 of the book
# and in the noteboo ggtogg.ipynb
#


#
def V3gtchannel(p1, hp1, p2, hp2, p3, hp3, p4, hp4):
    r1 = V3gInOut(p1, hp1, p3, hp3)
    r2 = V3gInOut(p2, hp2, p4, hp4)
    result = dotprod4(r1, r2)
    return result


#
#
def V4gtchannel(p1, hp1, p2, hp2, p3, hp3, p4, hp4):
    eps1 = pol(p1, hp1)
    eps2 = pol(p2, hp2)
    eps3 = polbar(p3, hp3)
    eps4 = polbar(p4, hp4)
    r1 = dotprod4(eps1, eps4)
    r2 = dotprod4(eps2, eps3)
    r3 = dotprod4(eps1, eps2)
    r4 = dotprod4(eps3, eps4)
    result = -r1 * r2 + r3 * r4
    return result


#
def V3guchannel(p1, hp1, p2, hp2, p3, hp3, p4, hp4):
    r1 = V3gInOut(p1, hp1, p4, hp4)
    r2 = V3gInOut(p2, hp2, p3, hp3)
    result = dotprod4(r1, r2)
    return result


#
def V4guchannel(p1, hp1, p2, hp2, p3, hp3, p4, hp4):
    eps1 = pol(p1, hp1)
    eps2 = pol(p2, hp2)
    eps3 = polbar(p3, hp3)
    eps4 = polbar(p4, hp4)
    r1 = dotprod4(eps1, eps2)
    r2 = dotprod4(eps3, eps4)
    r3 = dotprod4(eps1, eps3)
    r4 = dotprod4(eps2, eps4)
    result = r1 * r2 - r3 * r4
    return result


#
#
def V3gschannel(p1, hp1, p2, hp2, p3, hp3, p4, hp4):
    r1 = V3gInIn(p1, hp1, p2, hp2)
    r2 = V3gOutOut(p3, hp3, p4, hp4)
    result = dotprod4(r1, r2)
    return result


#
def V4gschannel(p1, hp1, p2, hp2, p3, hp3, p4, hp4):
    eps1 = pol(p1, hp1)
    eps2 = pol(p2, hp2)
    eps3 = polbar(p3, hp3)
    eps4 = polbar(p4, hp4)
    r1 = dotprod4(eps1, eps3)
    r2 = dotprod4(eps2, eps4)
    r3 = dotprod4(eps1, eps4)
    r4 = dotprod4(eps2, eps3)
    result = r1 * r2 - r3 * r4
    return result


#
#
# The  following routines, e.e. V3gInIn re useful for certain checks in undertsanding
# the notebook ggtogg.ipynb
#
def V3gInIn(k, hk, p, hp):
    eps1 = pol(k, hk)
    eps2 = pol(p, hp)
    p1 = fourvec(k)
    p2 = fourvec(p)
    s1 = dotprod4(eps1, eps2)
    s2 = dotprod4(eps1, p2)
    s3 = dotprod4(eps2, p1)
    elm1 = s1 * (p1[0] - p2[0]) + 2 * s2 * eps2[0] - 2 * s3 * eps1[0]
    elm2 = s1 * (p1[1] - p2[1]) + 2 * s2 * eps2[1] - 2 * s3 * eps1[1]
    elm3 = s1 * (p1[2] - p2[2]) + 2 * s2 * eps2[2] - 2 * s3 * eps1[2]
    elm4 = s1 * (p1[3] - p2[3]) + 2 * s2 * eps2[3] - 2 * s3 * eps1[3]
    result = [elm1, elm2, elm3, elm4]
    return result


def V3gOutOut(k, hk, p, hp):
    eps3 = polbar(k, hk)
    eps4 = polbar(p, hp)
    p3 = fourvec(k)
    p4 = fourvec(p)
    s1 = dotprod4(eps3, eps4)
    s2 = dotprod4(eps3, p4)
    s3 = dotprod4(eps4, p3)
    elm1 = s1 * (p4[0] - p3[0]) - 2 * s2 * eps4[0] + 2 * s3 * eps3[0]
    elm2 = s1 * (p4[1] - p3[1]) - 2 * s2 * eps4[1] + 2 * s3 * eps3[1]
    elm3 = s1 * (p4[2] - p3[2]) - 2 * s2 * eps4[2] + 2 * s3 * eps3[2]
    elm4 = s1 * (p4[3] - p3[3]) - 2 * s2 * eps4[3] + 2 * s3 * eps3[3]
    result = [elm1, elm2, elm3, elm4]
    return result


def V3gInOut(k, hk, p, hp):
    eps1 = pol(k, hk)
    eps3 = polbar(p, hp)
    p1 = fourvec(k)
    p3 = fourvec(p)
    s1 = dotprod4(eps1, eps3)
    s2 = dotprod4(eps1, p3)
    s3 = dotprod4(eps3, p1)
    elm1 = simplify(s1 * (p1[0] + p3[0]) - 2 * s2 * eps3[0] - 2 * s3 * eps1[0])
    elm2 = simplify(s1 * (p1[1] + p3[1]) - 2 * s2 * eps3[1] - 2 * s3 * eps1[1])
    elm3 = simplify(s1 * (p1[2] + p3[2]) - 2 * s2 * eps3[2] - 2 * s3 * eps1[2])
    elm4 = simplify(s1 * (p1[3] + p3[3]) - 2 * s2 * eps3[3] - 2 * s3 * eps1[3])
    result = [elm1, elm2, elm3, elm4]
    return result


#
# And of cours we need the SU3 Generators
#
lam1 = Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
lam2 = Matrix([[0, -I, 0], [I, 0, 0], [0, 0, 0]])
lam3 = Matrix([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
lam4 = Matrix([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
lam5 = Matrix([[0, 0, -I], [0, 0, 0], [I, 0, 0]])
lam6 = Matrix([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
lam7 = Matrix([[0, 0, 0], [0, 0, -I], [0, I, 0]])
lam08 = Matrix([[1, 0, 0], [0, 1, 0], [0, 0, -2]])
lam8 = lam08 / sqrt(3)


#
def fsu3(i, j, k):
    """Calculate QCD structure constants
    by using permutations of know
    structure constants
    """
    sq3 = sqrt(3) / 2
    s12 = Rational(1, 2)
    lf = [
        ((0, 1, 2), 1),
        ((1, 0, 2), -1),
        ((0, 3, 6), s12),
        ((3, 0, 6), -s12),
        ((1, 3, 5), s12),
        ((3, 1, 5), -s12),
        ((1, 4, 6), s12),
        ((4, 1, 6), -s12),
        ((2, 3, 4), s12),
        ((3, 2, 4), -s12),
        ((4, 0, 5), s12),
        ((0, 4, 5), -s12),
        ((5, 2, 6), s12),
        ((2, 5, 6), -s12),
        ((3, 4, 7), sq3),
        ((4, 3, 7), -sq3),
        ((5, 6, 7), sq3),
        ((6, 5, 7), -sq3),
    ]
    t = (i, j, k)
    #
    for tt in lf:
        if not set(t) == set(tt[0]):
            continue
        if Permutation([t]) == Permutation([tt[0]]):
            return tt[1]
        else:
            continue
    return 0


####

## Marcio Modifications - new functions

def vbuwMB(p1, h1, p2, h2, qv, mmed, cv, ca):

    #boson_mass = qv[1]
    qvec = qv / mmed

    # GCW theory
    cv, ca, gz, gw, thetaw, gz_theta, gw_theta, gf = symbols(r'c_v c_a g_Z g_W theta_W g_Z_theta g_W_theta, g_f', real=True, )
    # gz = gw / cos(thetaw)
    gv = cv / 2
    ga = ca / 2
    #gz_theta = (gw / cos(thetaw))
    
    #gw_theta = gw * cos(thetaw)
    EWproj = -I * gf * (gv * one - ga * g5)


    j0 = -I * (vbar(p1, h1) * g0 * EWproj * u(p2, h2)  - vbar(p1, h1) * g0 * EWproj * qvec[0]  * u(p2, h2) )
    j1 = -I * (vbar(p1, h1) * g1 * EWproj * u(p2, h2)  - vbar(p1, h1) * g1 * EWproj * qvec[1]  * u(p2, h2) )
    j2 = -I * (vbar(p1, h1) * g2 * EWproj * u(p2, h2)  - vbar(p1, h1) * g2 * EWproj * qvec[2]  * u(p2, h2) )
    j3 = -I * (vbar(p1, h1) * g3 * EWproj * u(p2, h2)  - vbar(p1, h1) * g3 * EWproj * qvec[3]  * u(p2, h2) )
    result = Matrix([simplify(j0[0]), simplify(j1[0]), simplify(j2[0]), simplify(j3[0])]) 
    return result



def V3ZOutOut(k, hk, p, hp, qv, mmed, cv, ca):
    
    cv, ca, gz, gw, thetaw, gz_theta, gw_theta, gv = symbols(r'c_v c_a g_Z g_W theta_W g_Z_theta g_W_theta, g_v', real=True)

    eps3 = polbar(k, hk)
    eps4 = polbar(p, hp)
    
    p3 = fourvec(k)
    p4 = fourvec(p)

    #boson_mass = qv[1]
    qvec = qv / mmed


    s1 = dotprod4(eps3, eps4) # 0
    s2 = dotprod4(eps3, p4)   # GeV
    s3 = dotprod4(eps4, p3)   # GeV

    elm1 = s1 * (p4[0] - p3[0]) - 2 * s2 * eps4[0] + 2 * s3 * eps3[0]  ## GeV
    elm2 = s1 * (p4[1] - p3[1]) - 2 * s2 * eps4[1] + 2 * s3 * eps3[1]  ## GeV
    elm3 = s1 * (p4[2] - p3[2]) - 2 * s2 * eps4[2] + 2 * s3 * eps3[2]  ## GeV
    elm4 = s1 * (p4[3] - p3[3]) - 2 * s2 * eps4[3] + 2 * s3 * eps3[3]  ## GeV
    
    w3 = dotprod4(qvec, p3)     # GeV 
    w4 = dotprod4(qvec, p4)     # GeV 
    we3 = dotprod4(qvec, eps3)  # 0
    we4 = dotprod4(qvec, eps4)  # 0 
    print(f'w3 {w3}')
    print(f'w4 {w4}')
    wd = w3 - w4  # = dotprod4(qvec, p4 - p3)

    SSelm1 = ((s1 * (w3 - w4))  - ((2 * s2) * we3)  +  ((2 * s3)  * we4) ) #* qvec[0]     ## GeV
    SSelm2 = ((s1 * (w3 - w4))  - ((2 * s2) * we3)  +  ((2 * s3)  * we4) ) #* qvec[0]     ## GeV
    SSelm3 = ((s1 * (w3 - w4))  - ((2 * s2) * we3)  +  ((2 * s3)  * we4) ) #* qvec[0]     ## GeV
    SSelm4 = ((s1 * (w3 - w4))  - ((2 * s2) * we3)  +  ((2 * s3)  * we4) ) #* qvec[0]     ## GeV

    


    result = Matrix([elm1 - SSelm1, elm2 - SSelm2, elm3 - SSelm3, elm4 - SSelm4]) * I * gv

    return result





def vbuwMB_noqq(p1, h1, p2, h2, qv, mmed, cv, ca):

    #boson_mass = qv[1]
    qvec = qv / mmed

    # GCW theory
    cv, ca, gz, gw, thetaw, gz_theta, gw_theta, gf = symbols(r'c_v c_a g_Z g_W theta_W g_Z_theta g_W_theta, g_f', real=True, )
    # gz = gw / cos(thetaw)
    gv = cv / 2
    ga = ca / 2
    #gz_theta = (gw / cos(thetaw))
    #gw_theta = gw * cos(thetaw)
    EWproj = -I * gf * (gv * one - ga * g5)


    j0 = -I * (vbar(p1, h1) * g0 * EWproj * u(p2, h2) ) #- vbar(p1, h1) * g0 * EWproj * qvec[0]  * u(p2, h2) )
    j1 = -I * (vbar(p1, h1) * g1 * EWproj * u(p2, h2) ) #- vbar(p1, h1) * g1 * EWproj * qvec[1]  * u(p2, h2) )
    j2 = -I * (vbar(p1, h1) * g2 * EWproj * u(p2, h2) ) #- vbar(p1, h1) * g2 * EWproj * qvec[2]  * u(p2, h2) )
    j3 = -I * (vbar(p1, h1) * g3 * EWproj * u(p2, h2) ) #- vbar(p1, h1) * g3 * EWproj * qvec[3]  * u(p2, h2) )
    result = Matrix([simplify(j0[0]), simplify(j1[0]), simplify(j2[0]), simplify(j3[0])]) 
    return result



def  V3ZOutOut_noqq(k, hk, p, hp, qv, mmed, cv, ca):
    
    cv, ca, gz, gw, thetaw, gz_theta, gw_theta, gv = symbols(r'c_v c_a g_Z g_W theta_W g_Z_theta g_W_theta, g_v', real=True)

    eps3 = polbar(k, hk)
    eps4 = polbar(p, hp)
    
    p3 = fourvec(k)
    p4 = fourvec(p)

    qvec = qv / mmed

    s1 = dotprod4(eps3, eps4) # 0
    s2 = dotprod4(eps3, p4)   # GeV
    s3 = dotprod4(eps4, p3)   # GeV

    elm1 = s1 * (p4[0] - p3[0]) - 2 * s2 * eps4[0] + 2 * s3 * eps3[0]  ## GeV
    elm2 = s1 * (p4[1] - p3[1]) - 2 * s2 * eps4[1] + 2 * s3 * eps3[1]  ## GeV
    elm3 = s1 * (p4[2] - p3[2]) - 2 * s2 * eps4[2] + 2 * s3 * eps3[2]  ## GeV
    elm4 = s1 * (p4[3] - p3[3]) - 2 * s2 * eps4[3] + 2 * s3 * eps3[3]  ## GeV
    
    w3 = dotprod4(qvec, p3)     # GeV 
    w4 = dotprod4(qvec, p4)     # GeV 
    we3 = dotprod4(qvec, eps3)  # 0
    we4 = dotprod4(qvec, eps4)  # 0 

    wd = w3 - w4  # = dotprod4(qvec, p4 - p3)

    # SSelm1 = ((s1 * (w3 - w4))  - ((2 * s2) * we3)  +  ((2 * s3)  * we4) ) #* qvec[0]     ## GeV
    # SSelm2 = ((s1 * (w3 - w4))  - ((2 * s2) * we3)  +  ((2 * s3)  * we4) ) #* qvec[0]     ## GeV
    # SSelm3 = ((s1 * (w3 - w4))  - ((2 * s2) * we3)  +  ((2 * s3)  * we4) ) #* qvec[0]     ## GeV
    # SSelm4 = ((s1 * (w3 - w4))  - ((2 * s2) * we3)  +  ((2 * s3)  * we4) ) #* qvec[0]     ## GeV

    result = Matrix([elm1 , elm2 , elm3 , elm4 ]) * I * gv
    #result = Matrix([elm1 - SSelm1, elm2 - SSelm2, elm3 - SSelm3, elm4 - SSelm4]) * I * gv

    return result



## !!! 
## Dark matter scenarios 

## F + F -> V
def vbu_VDM(p1, h1, p2, h2, qv, mmed, cv, ca, secterm = True):

    st = 1 if secterm else 0 

    #boson_mass = qv[1]
    qvec = qv / mmed

    gf, gl, gr, grv, gra, glv, gla = symbols(r'g_f g_l g_r g_{rv} g_{ra} g_{lv} g_{la}', real=True, )

    

    # GCW theory
    
    # gz = gw / cos(thetaw)
    #glv = cv / 2
    #gla = ca / 2
    #gz_theta = (gw / cos(thetaw))
    
    #gw_theta = gw * cos(thetaw)
    EWproj = -I * gf * (glv * one - gla * g5)
    Glr = - I * (gl * (glv * one - gla * g5)  + gr * (grv * one + gra * g5)) / 2
    
    ## EXPLANATION:
    # gr = 0 -> usual matter, no right handed stuff
    # gl = 0 -> exotic matter, only right handed stuff
    # gl = gr -> mixing, two things coexist
    # gl = - gr -> ???
    # glv, gla, gra, grv vector or axial vector stuff, given a right or left handed scenario, in a right handded scenario replaced by ca and cv, should give EW theory



    j0 = -I * (vbar(p1, h1) * g0 * Glr * u(p2, h2)  - st * vbar(p1, h1) * g0 * Glr * qvec[0]  * u(p2, h2) )
    j1 = -I * (vbar(p1, h1) * g1 * Glr * u(p2, h2)  - st * vbar(p1, h1) * g1 * Glr * qvec[1]  * u(p2, h2) )
    j2 = -I * (vbar(p1, h1) * g2 * Glr * u(p2, h2)  - st * vbar(p1, h1) * g2 * Glr * qvec[2]  * u(p2, h2) )
    j3 = -I * (vbar(p1, h1) * g3 * Glr * u(p2, h2)  - st * vbar(p1, h1) * g3 * Glr * qvec[3]  * u(p2, h2) )
    result = Matrix([simplify(j0[0]), simplify(j1[0]), simplify(j2[0]), simplify(j3[0])]) 
    return result



def V3ZOutOut_VDM(k, hk, p, hp, qv, mmed, cv, ca, secterm = True):
    
    cv, ca, gz, gw, thetaw, gz_theta, gw_theta, gv = symbols(r'c_v c_a g_Z g_W theta_W g_Z_theta g_W_theta, g_v', real=True)

    st = 1 if secterm else 0 

    eps3 = polbar(k, hk)
    eps4 = polbar(p, hp)
    
    p3 = fourvec(k)
    p4 = fourvec(p)

    #boson_mass = qv[1]
    qvec = qv / mmed


    s1 = dotprod4(eps3, eps4) # 0
    s2 = dotprod4(eps3, p4)   # GeV
    s3 = dotprod4(eps4, p3)   # GeV

    elm1 = s1 * (p4[0] - p3[0]) - 2 * s2 * eps4[0] + 2 * s3 * eps3[0]  ## GeV
    elm2 = s1 * (p4[1] - p3[1]) - 2 * s2 * eps4[1] + 2 * s3 * eps3[1]  ## GeV
    elm3 = s1 * (p4[2] - p3[2]) - 2 * s2 * eps4[2] + 2 * s3 * eps3[2]  ## GeV
    elm4 = s1 * (p4[3] - p3[3]) - 2 * s2 * eps4[3] + 2 * s3 * eps3[3]  ## GeV
    
    w3 = dotprod4(qvec, p3)     # GeV 
    w4 = dotprod4(qvec, p4)     # GeV 
    we3 = dotprod4(qvec, eps3)  # 0
    we4 = dotprod4(qvec, eps4)  # 0 
    print(f'w3 {w3}')
    print(f'w4 {w4}')
    wd = w3 - w4  # = dotprod4(qvec, p4 - p3)

    SSelm1 = st * ((s1 * (w3 - w4))  - ((2 * s2) * we3)  +  ((2 * s3)  * we4) ) #* qvec[0]     ## GeV
    SSelm2 = st * ((s1 * (w3 - w4))  - ((2 * s2) * we3)  +  ((2 * s3)  * we4) ) #* qvec[0]     ## GeV
    SSelm3 = st * ((s1 * (w3 - w4))  - ((2 * s2) * we3)  +  ((2 * s3)  * we4) ) #* qvec[0]     ## GeV
    SSelm4 = st * ((s1 * (w3 - w4))  - ((2 * s2) * we3)  +  ((2 * s3)  * we4) ) #* qvec[0]     ## GeV

    


    result = Matrix([elm1 - SSelm1, elm2 - SSelm2, elm3 - SSelm3, elm4 - SSelm4]) * I * gv

    return result
                


## F + F -> S
def ff_SDM_(p1, h1, p2, h2, qv, mmed):

    #st = 1 if secterm else 0 

    #mediator mass = qv[1]
    qvec = qv / mmed

    gf, gs = symbols(r'g_f g_s', real=True)

    

    # GCW theory
    
    # gz = gw / cos(thetaw)
    #glv = cv / 2
    #gla = ca / 2
    #gz_theta = (gw / cos(thetaw))
    
    #gw_theta = gw * cos(thetaw)
    #EWproj = -I * gf * (glv * one - gla * g5)
    GS =  - I * (gs) # * (glv * one - gla * g5)  + grx * (grv * one + gra * g5)) / 2
    
    ## EXPLANATION:
    # gr = 0 -> usual matter, no right handed stuff
    # gl = 0 -> exotic matter, only right handed stuff
    # gl = gr -> mixing, two things coexist
    # gl = - gr -> ???
    # glv, gla, gra, grv vector or axial vector stuff, given a right or left handed scenario, in a right handded scenario replaced by ca and cv, should give EW theory



    j0 =  (vbar(p1, h1) * GS * u(p2, h2)) #* I #- st * vbar(p1, h1) * g0 * Glr * qvec[0]  * u(p2, h2) )
    j1 =  (vbar(p1, h1) * GS * u(p2, h2)) #* I #- st * vbar(p1, h1) * g1 * Glr * qvec[1]  * u(p2, h2) )
    j2 =  (vbar(p1, h1) * GS * u(p2, h2)) #* I #- st * vbar(p1, h1) * g2 * Glr * qvec[2]  * u(p2, h2) )
    j3 =  (vbar(p1, h1) * GS * u(p2, h2)) #* I #- st * vbar(p1, h1) * g3 * Glr * qvec[3]  * u(p2, h2) )
    result = Matrix([simplify(j0[0]), simplify(j1[0]), simplify(j2[0]), simplify(j3[0])]) 
    return result


# S -> F + -F 
def _SDM_ff(p1, h1, p2, h2, qv, mmed):

    #st = 1 if secterm else 0 

    #mediator mass = qv[1]
    qvec = qv / mmed

    gf, gsx = symbols(r'g_f g_sx', real=True )

    

    GS =  - I * (gsx) # * (glv * one - gla * g5)  + grx * (grv * one + gra * g5)) / 2
    
    ## EXPLANATION:
    # gr = 0 -> usual matter, no right handed stuff
    # gl = 0 -> exotic matter, only right handed stuff
    # gl = gr -> mixing, two things coexist
    # gl = - gr -> ???
    # glv, gla, gra, grv vector or axial vector stuff, given a right or left handed scenario, in a right handded scenario replaced by ca and cv, should give EW theory



    j0 = (vbar(p1, h1) * GS * u(p2, h2))
    j1 = (vbar(p1, h1) * GS * u(p2, h2)) 
    j2 = (vbar(p1, h1) * GS * u(p2, h2)) 
    j3 = (vbar(p1, h1) * GS * u(p2, h2))
    result = Matrix([simplify(j0[0]), simplify(j1[0]), simplify(j2[0]), simplify(j3[0])])
    return result



def phi(pp, ha):
    res = Matrix([[1], [1], [1], [1]])
    return res

def phibar(pp, ha):
    res = Matrix([[1, 1, 1, 1]])
    return res



def Amplitude_schannel(u1, p1, u2, p2, prp, u3, p3, u4, p4):


    ## Breit-Wigner denominator, to be added in the end of the calculation for simplicity. 
    spinor = {
            'u'         : {'type': 'F', 'fn': u},
            'v'         : {'type': 'F', 'fn': v},
            'vbar'      : {'type': 'F', 'fn': vbar},
            'ubar'      : {'type': 'F', 'fn': ubar},
            'pol'       : {'type': 'V', 'fn': pol},
            'polbar'    : {'type': 'V', 'fn': polbar},	
            'phi'       : {'type': 'S', 'fn': phi},
            'phibar'    : {'type': 'S', 'fn': phibar}
            }
    
    # prop = {
    #         'scalar'    : 'S',
    #         'vector'    : 'V',
    #         'axial'     : 'A',
    #         'pseudoscalar' : 'P',
    #         'fermion' : 'F',
    # }
    
    ## Breit-Wigner denominator, to be added in the end of the calculation for simplicity. 
    def get_propagator(prop, type1, dm = False):

        # pick the correct indice for SM or DM mechanics
        dmsm = r"\chi" if dm else r"\psi"

        s, Mmed, Gamma    =  symbols(r's M_{med} Gamma', real=True )
        gs, gps           =  symbols(f'g_s{dmsm} g_ps{dmsm}', real=True )
        gv, gav           =  symbols(f'g_v{dmsm} g_av{dmsm}', real=True )
        #gs, gps, gsx =  symbols(r'g_{v} g_{ps} g_{sx}', real=True )

        denom = (s - Mmed**2 + I * (Mmed * Gamma))
        
        if prop == 'scalar':
            vertex = {
                        f"F" : -I * ( gs * one + I * gps * g5 ), ## scalar and pseudoscalar couplings
                        f"V" : -I *  gs * one,                  ## Scalar vector coupling
                        f"S" : -I *  gs * one,                  ## Triple scalar simple coupling
                     }                  

            propagator = -1 / denom

        elif prop == 'vector':
            # vertex = {
            #             f"{type1}{type2}": -I * ( gs * one + I * gps * g5 ), ## scalar and pseudoscalar couplings
            #             f"{type1}{type2}": -I *  gsx * one,
            #             f"{type1}{type2}": -I *  gsx * one,
            #          }                  ## Triple scalar simple coupling
            pass

        elif prop == 'fermion':
            print('Not yet')
            pass
    
        return vertex[type1], propagator
    

    def init_J():
        hel_fermion = [-1 , 1 ]
        vertex, propagator = get_propagator(prp, spinor[u1]['type']) ## Get initial propagator
        Jc_init = [] ## define initial current 
        for hl in hel_fermion:
            for hr in hel_fermion:    
                jc = []
                for i in range(4):
                    #print(spinor[u1](p1, hl))
                    #print('newspinor')
                    ji = spinor[u1]['fn'](p1, hl) * vertex * propagator *  spinor[u2]['fn'](p2, hr)   ## IMPORTANT!!! intial current - propagator only here
                    jc.append(simplify(ji[0]))

                Jc_init.append(Matrix(jc))

        return Jc_init

    def final_J(types):
        hel_fermion = [-1 , 1 ] if  types != 'V' else [-1 ,0, 1 ]
        vertex, propagator = get_propagator(prp, types, dm=True) ## Get initial propagator
        Jc_final = [] ## define initial current 
        for hl in hel_fermion:
            for hr in hel_fermion:    
                jc = []
                for i in range(4):
                    #print(spinor[u3](p3, hl))
                    jf = spinor[u3]['fn'](p3, hl) * vertex  * spinor[u4]['fn'](p4, hr)  ## IMPORTANT!!! # second current
                    #print(jf)
                    jc.append(simplify(jf[0]))

                Jc_final.append(Matrix(jc))

        return Jc_final
    
    ## Trace final result
    Ji = init_J()
    Jf = final_J(spinor[u4]['type'])
    Terms = []
    for j_init in Ji:
        for j_final in Jf:
            #print('append term')

            Terms.append(simplify(dotprod4(j_init , j_final)))
    

    return Terms


def decaimento_Gamma(prp, u3, p3, u4, p4):


    ## Breit-Wigner denominator, to be added in the end of the calculation for simplicity. 
    spinor = {
            'u'         : {'type': 'F', 'fn': u},
            'v'         : {'type': 'F', 'fn': v},
            'vbar'      : {'type': 'F', 'fn': vbar},
            'ubar'      : {'type': 'F', 'fn': ubar},
            'pol'       : {'type': 'V', 'fn': pol},
            'polbar'    : {'type': 'V', 'fn': polbar},	
            'phi'       : {'type': 'S', 'fn': phi},
            'phibar'    : {'type': 'S', 'fn': phibar}
            }
    
    # prop = {
    #         'scalar'    : 'S',
    #         'vector'    : 'V',
    #         'axial'     : 'A',
    #         'pseudoscalar' : 'P',
    #         'fermion' : 'F',
    # }
    
    ## Breit-Wigner denominator, to be added in the end of the calculation for simplicity. 
    def get_propagator(prop, types, dm = False):

        # pick the correct indice for SM or DM mechanics
        dmsm = r"\chi" if dm else r"\psi"

        s, Mmed, Gamma    =  symbols(r's M_{med} Gamma', real=True )
        gs, gps           =  symbols(f'g_s{dmsm} g_ps{dmsm}', real=True )
        gv, gav           =  symbols(f'g_v{dmsm} g_av{dmsm}', real=True )
        #gs, gps, gsx =  symbols(r'g_{v} g_{ps} g_{sx}', real=True )

        denom = (s - Mmed**2 + I * (Mmed * Gamma))
        
        if prop == 'scalar':
            vertex = {
                        f"F" : I * -I * ( gs * one),# + I * gps * g5 ), ## scalar and pseudoscalar couplings
                        f"V" : I * gs * one,                  ## Scalar vector coupling
                        f"S" : -I *  gs * one,                  ## Triple scalar simple coupling
                     }                  

            propagator = -1 / denom

        elif prop == 'vector':
            # vertex = {
            #             f"{type1}{type2}": -I * ( gs * one + I * gps * g5 ), ## scalar and pseudoscalar couplings
            #             f"{type1}{type2}": -I *  gsx * one,
            #             f"{type1}{type2}": -I *  gsx * one,
            #          }                  ## Triple scalar simple coupling
            pass

        elif prop == 'fermion':
            print('Not yet')
            pass
    
        return vertex[types], propagator
    

    def init_J(types):
        #print(types)
        hel_fermion = [-1 , 1 ] if  types != 'V' else [-1 ,0, 1 ]
        vertex, propagator = get_propagator(prp, types) ## Get initial propagator
        Jc_init = [] ## define initial current 
        for hl in hel_fermion: 
            jc = []
            for i in range(4):
                #print(hl)
                #print(spinor[u1](p1, hl))
                #print('newspinor')
                ji =  spinor[u3]['fn'](p3, hl) * vertex  
                jc.append(simplify(ji[0]))

            Jc_init.append(Matrix(jc))

        return Jc_init

    def final_J(types):
        #print(types)
        hel_fermion = [-1 , 1 ] if  types != 'V' else [-1 ,0, 1 ]
        vertex, propagator = get_propagator(prp, types, dm=True) ## Get initial propagator
        Jc_final = [] ## define initial current 
        
        for hl in hel_fermion:  
            jc = []

            for i in range(4):
                #print(hl)
                #print(spinor[u3](p3, hl))
                jf =   one * spinor[u4]['fn'](p4, hl)   
                #print(jf)
                jc.append(simplify(jf[0]))

            Jc_final.append(Matrix(jc))

        return Jc_final
    
    ## Trace final result
    Ji = init_J(spinor[u3]['type'])
    Jf = final_J(spinor[u4]['type'])
    Terms = []
    for j_init in Ji:
        for j_final in Jf:
            #print('append term')

            Terms.append(simplify(dotprod4(j_init , j_final)))
    

    return Terms




print("Done")

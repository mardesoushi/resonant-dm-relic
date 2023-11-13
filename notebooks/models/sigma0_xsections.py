import numpy as np
from numpy import sqrt as sqrt

# ------------------------------------------------------------------------------
# DECAY RATES
# ------------------------------------------------------------------------------

@np.vectorize
def GMSM(s, mq, Mmed, mx, gr, gl, gx, Nf):
    if Mmed < 2*mx:
        return np.nan
    M = Mmed
    Gamma_SM = (np.sqrt(1 - 4*(mq**2)/(M**2)) * ((gl**2)*(M**2 - mq**2) + 6*gl*gr*(mq**2) + (gr**2)*((M**2) - (mq**2))))/(24*np.pi*M)  ## x9 for all SM fermions
    
    return Nf*Gamma_SM

@np.vectorize
def GMS(s, mq, Mmed, mx, gr, gl, gx, Nf):
    if Mmed < 2*mx:
        return np.nan
    M = Mmed
    
    gxs = gx
    
    Gamma_SDM = ((gxs**2)*np.sqrt(1 - 4*(mx**2)/(M**2))*((M**2) - 4*(mx**2)))/(48*np.pi*M)
    return Gamma_SDM

@np.vectorize
def GMF(s, mq, Mmed, mx, gr, gl, gx, Nf):
    if Mmed < 2*mx:
        return np.nan
    M = Mmed
    grx = glx = gx

    Gamma_DM = np.sqrt(1 - 4*mx**2/M**2)*(glx**2*(M**2 - mx**2) + 6*glx*grx*mx**2 + grx**2*(M**2 - mx**2))/(24*np.pi*M)
    
    return Gamma_DM


@np.vectorize
def GMV(s, me, Mmed, mx, gr, gl, gx, Nf):
    if Mmed < 2*mx:
        return np.nan
    
    m_f = mx 
    M_med = Mmed 
    
    #result=  (1/192)*gx**2  * (M_med**2 - 4*m_f**2)**(3/2)  / (np.pi*m_f**2)

    result=  ((1/192) * gx**2  * (M_med**(3/2)) * ( (1 - ((4*m_f**2) / (M_med**2)))**(3/2) )) / (np.pi*m_f**2)
    return result


# @np.vectorize
# def GMV(s, me, Mmed, mx, gr, gl, gx):
#     if Mmed < 2*mx:
#          return np.nan   
    
#     t1 = (1/(192*np.pi) ) *  Mmed  *  gx**2 
#     t2 = (1 - ((4*mx**2)/(Mmed**2))**(3/2) )

#     res = t1 * t2

#     return res 
# @np.vectorize
# def GMV(s, mq, Mmed, mx, gr, gl, gx):


#     Gamma_VDM = gxv**2 * np.sqrt(1 - 4*mx**2/M**2) * (5*M**2 - 32*mx**2)/(96*np.pi*M)
#     return Gamma_VDM
class GM:
    def __init__(self,  DMname):
    
        GMdecay = {"Scalar":GMS, 
                "Fermion":GMF, 
                "Vector":GMV,
                "SM":GMSM}
        self.DMname = DMname
        self.GMdecay = GMdecay[self.DMname]



################################################################################ 
# ------------------------------------------------------------------------------
# TOTAL CROSS SECTIONS
# ------------------------------------------------------------------------------
################################################################################        
# ------------------------------------------------------------------------------
# SCALAR DARK MATTER
# ------------------------------------------------------------------------------
@np.vectorize
def sig0_S(s, mq, Mmed, mx, gr, gl, gx, Nf):    ## integrated total cross section \sigma_0 (ee -> Z' -> XX)

    if s < 4*(mx**2): ##Define the correct physical domain for s channel 
        return np.nan  
    
    else:
        M = Mmed
        #coupling control

        gxs = gx


        dmname = 'Scalar'
        Bi = GM('SM').GMdecay( s, mq, Mmed, mx, gr, gl, gx, Nf)
        Bf = GM(dmname).GMdecay( s, mq, Mmed, mx, gr, gl, gx, Nf)
        Gamma = Bi + Bf
        
        #totalCS = gxs**2*np.sqrt(-4*mq**2 + s)*( (np.sqrt((-4*mx**2 + s)))*(-4*mx**2 + s))*(-mq**2*(gl**2 - 6*gl*gr + gr**2) + s*(gl**2 + gr**2))/(96*np.pi*s**2*(Gamma**2*M**2 + (M**2 - s)**2))
        #New try
        totalCS = -gxs**2*np.sqrt(-(4*mq**2 - s)*(-4*mx**2 + s))*(-4*mx**2 + s)*(gl**2*(mq**2 - s) - 6*gl*gr*mq**2 + gr**2*(mq**2 - s))/(192*np.pi*s**2*(M**4 + M**2*(Gamma**2 - 2*s) + s**2))
        return totalCS

# ------------------------------------------------------------------------------
# FERMION DARK MATTER
# ------------------------------------------------------------------------------
@np.vectorize
def sig0_F(s, mq, Mmed, mx, gr, gl, gx, Nf):  ## integrated total cross section FERMION \sigma_0 (ee -> Z' -> XX)

    if s < 4*(mx**2): ##Define the correct physical domain for s channel 
        return np.nan  
    
    else:
        M = Mmed
        
        #coupling control
        glx = gx
        grx = gx
        #----------------
        
        dmname = 'Fermion'
        Bi = GM('SM').GMdecay(s, mq, Mmed, mx, gr, gl, gx, Nf)
        Bf = GM(dmname).GMdecay( s, mq, Mmed, mx, gr, gl, gx, Nf)
        Gamma = Bi + Bf

        
        totalCS = np.sqrt(-4*mq**2 + s)*np.sqrt(-4*mx**2 + s)*(gl**2*(4*M**4*mq**2*mx**2*(glx**2 - 3*glx*grx + grx**2) - M**2*s*(M**2*mq**2*(glx**2 - 6*glx*grx + grx**2) + mx**2*(M**2*(glx**2 + grx**2) + 6*mq**2*(glx - grx)**2)) + s**2*(M**4*(glx**2 + grx**2) + 3*mq**2*mx**2*(glx - grx)**2)) + 6*gl*gr*mx**2*(-2*M**4*mq**2*(glx**2 - 4*glx*grx + grx**2) + M**4*s*(glx**2 + grx**2) + 2*M**2*mq**2*s*(glx - grx)**2 - mq**2*s**2*(glx - grx)**2) + gr**2*(4*M**4*mq**2*mx**2*(glx**2 - 3*glx*grx + grx**2) - M**2*s*(M**2*mq**2*(glx**2 - 6*glx*grx + grx**2) + mx**2*(M**2*(glx**2 + grx**2) + 6*mq**2*(glx - grx)**2)) + s**2*(M**4*(glx**2 + grx**2) + 3*mq**2*mx**2*(glx - grx)**2)))/(48*np.pi*M**4*s**2*(Gamma**2*M**2 + (M**2 - s)**2))
        #Try this
        #totalCS = np.sqrt((-4*mq**2 + s)*(-4*mx**2 + s))*(gl**2*(glx**2*(M**4*(mq**2*(4*mx**2 - s) + s*(-mx**2 + s)) - 6*M**2*mq**2*mx**2*s + 3*mq**2*mx**2*s**2) - 6*glx*grx*mq**2*(M**4*(2*mx**2 - s) - 2*M**2*mx**2*s + mx**2*s**2) + grx**2*(M**4*(mq**2*(4*mx**2 - s) + s*(-mx**2 + s)) - 6*M**2*mq**2*mx**2*s + 3*mq**2*mx**2*s**2)) - 6*gl*gr*mx**2*(glx**2*(M**4*(2*mq**2 - s) - 2*M**2*mq**2*s + mq**2*s**2) - 2*glx*grx*mq**2*(4*M**4 - 2*M**2*s + s**2) + grx**2*(M**4*(2*mq**2 - s) - 2*M**2*mq**2*s + mq**2*s**2)) + gr**2*(glx**2*(M**4*(mq**2*(4*mx**2 - s) + s*(-mx**2 + s)) - 6*M**2*mq**2*mx**2*s + 3*mq**2*mx**2*s**2) - 6*glx*grx*mq**2*(M**4*(2*mx**2 - s) - 2*M**2*mx**2*s + mx**2*s**2) + grx**2*(M**4*(mq**2*(4*mx**2 - s) + s*(-mx**2 + s)) - 6*M**2*mq**2*mx**2*s + 3*mq**2*mx**2*s**2)))/(96*np.pi*M**4*s**2*(M**4 + M**2*(Gamma**2 - 2*s) + s**2))
    return totalCS

# ------------------------------------------------------------------------------
# VECTOR DARK MATTER
# ------------------------------------------------------------------------------
# @np.vectorize
# def sig0_V(s, mq, Mmed, mx, gr, gl, gx):    ## seção de choque e integração de "-t" a "+t"   

#     if s < 4*(mx**2): ##Define the correct physical domain for s channel 
#         return np.nan  
    
#     else:
#         M = Mmed
#         #coupling control
#         gxv = gx
#         dmname = 'Vector'
#         Bi = GM('SM').GMdecay(s, mq, Mmed, mx, gr, gl, gx)
#         Bf = GM(dmname).GMdecay(s, mq, Mmed, mx, gr, gl, gx)
#         Gamma = Bi + Bf

#         coup = gxv**2 * (gl**2 + gr**2) 
#         num1 = np.sqrt(s*(s - 4*mx**2))
#         num2 = -48*mx**6 - 68*mx**4 * s + 16*mx**2 * s**2  + s**3


#         numerator = coup * num1 * num2

#         denom = 768* np.pi * mx**4 * s * ((s - M**2)**2  + Gamma**2 * M ** 2) 

#         totalCS = numerator / denom


#         return totalCS

@np.vectorize
def sig0_V(s, mq, Mmed, mx, gr, gl, gx, Nf):

    M_med = Mmed
    m_f = mx 
    g_dm = gx 
    g_l = gl
    g_r = gr

    if s < 4*(mx**2): ##Define the correct physical domain for s channel 
        return np.nan      
    
    dmname = 'Vector'
    Bi = GM('SM').GMdecay(s, mq, Mmed, mx, gr, gl, gx, Nf)
    Bf = GM(dmname).GMdecay(s, mq, Mmed, mx, gr, gl, gx, Nf)
    Gamma = Bi + Bf    
    
    result = (1/192)*g_dm**2*(g_l**2 + g_r**2)*sqrt(-4*m_f**2 + s)*(-48*M_med**2*m_f**6 - 68*M_med**2*m_f**4*s + 16*M_med**2*m_f**2*s**2 + M_med**2*s**3 + 192*m_f**4*s**2 - 96*m_f**2*s**3 + 12*s**4)/(M_med**2*m_f**4*sqrt(s)*(Gamma**2*M_med**2 + M_med**4 - 2*M_med**2*s + s**2))
    return result

    # if s < 4*(mx**2): ##Define the correct physical domain for s channel 
    #     return np.nan   
    
    #totalCS = gxv**2*np.sqrt(-4*mq**2 + s)*( (np.sqrt((-4*mx**2 + s)))*(-4*mx**2 + s))*(24*mq**2*mx**4*(gl**2 + 3*gl*gr + gr**2) + 4*mx**2*s*(-2*mq**2*(gl**2 - 15*gl*gr + gr**2) + 3*mx**2*(gl**2 + gr**2)) + s**3*(gl**2 + gr**2) + 2*s**2*(mq**2*(gl**2 + 3*gl*gr + gr**2) + 10*mx**2*(gl**2 + gr**2)))/(384*np.pi*mx**4*s**2*(Gamma**2*M**2 + (M**2 - s)**2))

    
    # Try this
    #totalCS = -gxv**2*np.sqrt(-(4*mq**2 - s)*(-4*mx**2 + s))*(gl**2*(mq**2 - s) - 6*gl*gr*mq**2 + gr**2*(mq**2 - s))*(-48*mx**6 - 68*mx**4*s + 16*mx**2*s**2 + s**3)/(768*np.pi*mx**4*s**2*(M**4 + M**2*(Gamma**2 - 2*s) + s**2))

########### BW Cross Section ######################
@np.vectorize
def BW_sigma_S( s, mq, Mmed, mx, gr, gl, gx, Nf):
    dmname = 'Scalar'
    Ss = 1/2 ## Fermion spin - initial state
    Jj = 1 ## Z' spin
    M = Mmed
    Gamma =  GM(dmname).GMdecay( s, mq, Mmed, mx, gr, gl, gx) + GM('SM').GMdecay( s, mq, Mmed, mx, gr, gl, gx, Nf)
    Bi = GM('SM').GMdecay( s, mq, Mmed, mx, gr, gl, gx, Nf) / Gamma
    Bf = GM(dmname).GMdecay( s, mq, Mmed, mx, gr, gl, gx, Nf) / Gamma
    cs = 16*np.pi*Bf*Bi*Gamma**2*M**2*(2*Jj + 1)/((2*Ss + 1)**2*(-4*mq**2 + s)*(Gamma**2*M**2 + (M**2 - s)**2))

    return cs

@np.vectorize
def BW_sigma_F( s, mq, Mmed, mx, gr, gl, gx, Nf):
    dmname = 'Fermion'
    Ss = 1/2 ## Fermion spin - initial state
    Jj = 1 ## Z' spin
    M = Mmed
    Gamma =  GM(dmname).GMdecay( s, mq, Mmed, mx, gr, gl, gx, Nf) + GM('SM').GMdecay( s, mq, Mmed, mx, gr, gl, gx, Nf)
    Bi = GM('SM').GMdecay( s, mq, Mmed, mx, gr, gl, gx, Nf) / Gamma
    Bf = GM(dmname).GMdecay( s, mq, Mmed, mx, gr, gl, gx, Nf) / Gamma
    cs = 16*np.pi*Bf*Bi*Gamma**2*M**2*(2*Jj + 1)/((2*Ss + 1)**2*(-4*mq**2 + s)*(Gamma**2*M**2 + (M**2 - s)**2))

    return cs

@np.vectorize
def BW_sigma_V( s, mq, Mmed, mx, gr, gl, gx, Nf):
    dmname = 'Vector'
    Ss = 1/2 ## Fermion spin - initial state
    Jj = 1 ## Z' spin
    M = Mmed
    Gamma =  GM(dmname).GMdecay( s, mq, Mmed, mx, gr, gl, gx, Nf) + GM('SM').GMdecay( s, mq, Mmed, mx, gr, gl, gx, Nf)
    Bi = GM('SM').GMdecay(s, mq, Mmed, mx, gr, gl, gx, Nf) / Gamma
    Bf = GM(dmname).GMdecay(s, mq, Mmed, mx, gr, gl, gx, Nf) / Gamma
    cs = 16*np.pi*Bf*Bi*Gamma**2*M**2*(2*Jj + 1)/((2*Ss + 1)**2*(-4*mq**2 + s)*(Gamma**2*M**2 + (M**2 - s)**2))

    return cs

###########################################################################################################################

dmnames = ["Scalar", "Fermion", "Vector" ]

class SFV:
    def __init__(self,  DMname):
    
        sig0 = {"Scalar":sig0_S, 
                "Fermion":sig0_F, 
                "Vector":sig0_V,
                'BW_Scalar':BW_sigma_S,
                'BW_Fermion':BW_sigma_F,
                'BW_Vector':BW_sigma_V}
        self.DMname = DMname
        self.sig0 = sig0[self.DMname]




















































## not used ##

# ########### BW Cross Section ######################

# def BW_sigma_S( s, mq, Mmed, mx, gr, gl, gx):
#     dmname = 'Scalar'
#     Ss = 1/2 ## Fermion spin - initial state
#     Jj = 1 ## Z' spin
#     M = Mmed
#     Gamma =  GM(dmname).GMdecay( s, mq, Mmed, mx, gr, gl, gx) + GM('SM').GMdecay( s, mq, Mmed, mx, gr, gl, gx)
#     Bi = GM('SM').GMdecay( s, mq, Mmed, mx, gr, gl, gx) / Gamma
#     Bf = GM(dmname).GMdecay( s, mq, Mmed, mx, gr, gl, gx) / Gamma
#     cs = 16*np.pi*Bf*Bi*Gamma**2*M**2*(2*Jj + 1)/((2*Ss + 1)**2*(-4*mq**2 + s)*(Gamma**2*M**2 + (M**2 - s)**2))

#     return cs

# def BW_sigma_F( s, mq, Mmed, mx, gr, gl, gx):
#     dmname = 'Fermion'
#     Ss = 1/2 ## Fermion spin - initial state
#     Jj = 1 ## Z' spin
#     M = Mmed
#     Gamma =  GM(dmname).GMdecay( s, mq, Mmed, mx, gr, gl, gx) + GM('SM').GMdecay( s, mq, Mmed, mx, gr, gl, gx)
#     Bi = GM('SM').GMdecay( s, mq, Mmed, mx, gr, gl, gx) / Gamma
#     Bf = GM(dmname).GMdecay( s, mq, Mmed, mx, gr, gl, gx) / Gamma
#     cs = 16*np.pi*Bf*Bi*Gamma**2*M**2*(2*Jj + 1)/((2*Ss + 1)**2*(-4*mq**2 + s)*(Gamma**2*M**2 + (M**2 - s)**2))

#     return cs

# def BW_sigma_V( s, mq, Mmed, mx, gr, gl, gx):
#     dmname = 'Vector'
#     Ss = 1/2 ## Fermion spin - initial state
#     Jj = 1 ## Z' spin
#     M = Mmed
#     Gamma =  GM(dmname).GMdecay( s, mq, Mmed, mx, gr, gl, gx) + GM('SM').GMdecay( s, mq, Mmed, mx, gr, gl, gx)
#     Bi = GM('SM').GMdecay(s, mq, Mmed, mx, gr, gl, gx) / Gamma
#     Bf = GM(dmname).GMdecay(s, mq, Mmed, mx, gr, gl, gx) / Gamma
#     cs = 16*np.pi*Bf*Bi*Gamma**2*M**2*(2*Jj + 1)/((2*Ss + 1)**2*(-4*mq**2 + s)*(Gamma**2*M**2 + (M**2 - s)**2))

#     return cs

# ###########################################################################################################################

# #dmnames = ["Scalar", "Fermion", "Vector" ]
# dmnames = ["Scalar", "Fermion"]

# class SFV:
#     def __init__(self,  DMname):
    
#         sig0 = {"Scalar":sig0_S, 
#                 "Fermion":sig0_F, 
#                 "Vector":sig0_V,
#                 'BW_Scalar':BW_sigma_S,
#                 'BW_Fermion':BW_sigma_F,
#                 'BW_Vector':BW_sigma_V}
#         self.DMname = DMname
#         self.sig0 = sig0[self.DMname]





# # ------------------------------------------------------------------------------
# # FERMION DARK MATTER (dsig/dt)
# # ------------------------------------------------------------------------------
# @np.vectorize
# def sig0_F_dsdt( s, mq, Mmed, mx, gr, gl, gx):  ## integrated total cross section FERMION \sigma_0 (ee -> Z' -> XX)

#     M = Mmed
    
#     #coupling control
#     grx = glx = gx
#     #----------------
#     u = mq**2  + mx**2

#     Gamma_SM = 9* (np.sqrt(-4*mq**2 + s)*(gl**2*(-mq**2 + s) + 6*gl*gr*mq**2 + gr**2*(-mq**2 + s))/(24*np.pi*M**2)) ## x9 for all SM fermions
#     Gamma_DM = np.sqrt(-4*mx**2 + s)*(glx**2*(-mx**2 + s) + 6*glx*grx*mx**2 + grx**2*(-mx**2 + s))/(24*np.pi*M**2)
#     Gamma =  Gamma_DM + Gamma_SM

#     def cs(t):
#         return t*(gl**2*(glx**2*(M**4*(t**2 - 3*t*(mq**2 + mx**2 - s) + 3*(mq**2 + mx**2 - s)**2) - 6*M**2*mq**2*mx**2*s + 3*mq**2*mx**2*s**2) + 6*glx*grx*mq**2*(M**4*(-2*mx**2 + s) + 2*M**2*mx**2*s - mx**2*s**2) + grx**2*(M**4*(t**2 - 3*t*(mq**2 + mx**2) + 3*(mq**2 + mx**2)**2) - 6*M**2*mq**2*mx**2*s + 3*mq**2*mx**2*s**2)) + 6*gl*gr*mx**2*(-2*M**4*mq**2*(glx**2 - 4*glx*grx + grx**2) + M**4*s*(glx**2 + grx**2) + 2*M**2*mq**2*s*(glx - grx)**2 - mq**2*s**2*(glx - grx)**2) + gr**2*(glx**2*(M**4*(t**2 - 3*t*(mq**2 + mx**2) + 3*(mq**2 + mx**2)**2) - 6*M**2*mq**2*mx**2*s + 3*mq**2*mx**2*s**2) + 6*glx*grx*mq**2*(M**4*(-2*mx**2 + s) + 2*M**2*mx**2*s - mx**2*s**2) + grx**2*(M**4*(t**2 - 3*t*(mq**2 + mx**2 - s) + 3*(mq**2 + mx**2 - s)**2) - 6*M**2*mq**2*mx**2*s + 3*mq**2*mx**2*s**2)))/(48*np.pi*M**4*s**2*(Gamma**2*M**2 + (M**2 - s)**2))


#     t = ( (np.sqrt(s)) * (np.sqrt(s-4*mx**2)) + 2.*u-s )/2.
#     sig0 = ( cs(-t) - cs(t) )
#     return sig0

# @np.vectorize
# def sig0_SM( s, mq, Mmed, mx, gr, gx):  ## integrated total cross section FERMION \sigma_0 (ee -> Z' -> XX)    
#     M = Mmed

#     Gamma = 2.441404E-3 # For Z decay, PDG

#     totalCS = 1/192 * ( ( ( ca )**( 2 ) + ( cv )**( 2 ) ) )**( 2 ) * ( gz )**( 4 ) * ( np.pi )**( -1 ) * s * ( ( ( ( ( M )**( 2 ) + -1 * s ) )**( 2 ) + ( M )**( 2 ) * ( Gamma )**( 2 ) ) )**( -1 )

#     return totalCS


# @np.vectorize
# def sig0_CMS( s, mq, Mmed, mx, gr, gl, gx):
    
#     M = Mmed
#     #coupling control
#     grx = glx = gx
#     #----------------
#     Gamma_SM = 9* (np.sqrt(-4*mq**2 + s)*(gl**2*(-mq**2 + s) + 6*gl*gr*mq**2 + gr**2*(-mq**2 + s))/(24*np.pi*M**2)) ## x9 for all SM fermions
#     Gamma_DM = np.sqrt(-4*mx**2 + s)*(glx**2*(-mx**2 + s) + 6*glx*grx*mx**2 + grx**2*(-mx**2 + s))/(24*np.pi*M**2)
#     Gamma = Gamma_SM + Gamma_DM
    
#     totalCS = gr**2*gx**2*(mq**2 + 2*mx**2)*np.sqrt(-mq**2/mx**2 + 1)/(2*np.pi*(Gamma**2*M**2 + (M**2 - s)**2))
#     return totalCS


# def sig0_CMSav( s, mq, Mmed, mx, gr, gl, gx):
    
#     M = Mmed
#     #coupling control
#     grx = glx = gx
#     #----------------
#     Gamma_SM = 9*(1/24 * ( ( gl )**( 2 ) + ( gr )**( 2 ) ) * ( M )**( -2 ) * ( np.pi )**( -1 ) * ( s )**( 3/2 )) ## x9 for all SM fermions
#     Gamma_DM = 1/24 * ( M )**( -2 ) * ( np.pi )**( -1 ) * ( ( -4 * ( mx )**( 2 ) + s ) )**( 1/2 ) * ( 6 * glx * grx * ( mx )**( 2 ) + ( ( glx )**( 2 ) * ( -1 * ( mx )**( 2 ) + s ) + ( grx )**( 2 ) * ( -1 * ( mx )**( 2 ) + s ) ) )
#     Gamma = Gamma_SM + Gamma_DM
    
#     totalCS = gr**2*gx**2*mq**2*(-M**2 + 4*mx**2)**2*np.sqrt(-mq**2/mx**2 + 1)/(2*np.pi*M**4*(Gamma**2*M**2 + (M**2 - 4*mx**2)**2))
#     return totalCS


# def sig0_F_old( s, mq, Mmed, mx, gr, gl, gx):  ## integrated total cross section FERMION \sigma_0 (ee -> Z' -> XX)

#     mx = mx**2
#     u = mx + mq
#     Mmed = Mmed**2
#     g = (gl0**2) + (gr0**2)   # SM couplings
#     glr = 2*gl0*gr0

#     grx = gx0                # DM couplings
#     glx =gx0
    
#     gxf = (glx**2) + (grx**2)
#     glrx = 2*glx*grx
#     GM_SMF = (((np.sqrt(s)))/(48.*np.pi*Mmed))    * (2*g*(s) + (g*s)/2)   ## SM fermion decay
#     GM_F = (((np.sqrt(s-4.*mx)))/(48.*np.pi*Mmed))    * (2*gxf*(s-2*mx) + 3*glrx*mx +(gxf*s)/2) #+  GM_SMF ## decay width Z' -> XX (TO DO: add to this the another possible decays of Z' -> SM)
#     def cs(t):
#         try:
#             pre =  ((1./(16.*np.pi)) * (1/(s))) * (mx/(((s-Mmed)**2) + (Mmed*GM_F**2)) ) 
#             term1 =  (2./(s*mx)) * (((g**2)/3)*( ((s+t-u)**3) + ((t-u)**3) ) + 2.*g*glr*(s*u) )  
#             #term2 = (t/Mmed**2)* ( (g**2)*(s + 2.*u - 8.*mq*mx/s) - 4.*g*glr*(2.*s - u) + 4.*s*glr**2 ) 
#             #term3 = (8.*t/Mmed)*(g-glr)**2
#             term  = term1 #+ term2 + term3
#             cs = pre * term 
#         except ZeroDivisionError:
#             cs = 0
#         return cs
    
#     tp = mq**2 + mx**2 - s/2 + np.sqrt((mq**4 - 2*mq**2*(mq**2 + s) + (mq**2 - s)**2)/s)*np.sqrt((mx**4 - 2*mx**2*(mx**2 + s) + (mx**2 - s)**2)/s)/2
#     tm = mq**2 + mx**2 - s/2 - np.sqrt((mq**4 - 2*mq**2*(mq**2 + s) + (mq**2 - s)**2)/s)*np.sqrt((mx**4 - 2*mx**2*(mx**2 + s) + (mx**2 - s)**2)/s)/2   
#     #t = ( (np.sqrt(s)) * (np.sqrt(s-4*mx)) + 2.*u-s )/2.
#     sig0 = -( cs(tm)  - cs(tp) )
#     return sig0

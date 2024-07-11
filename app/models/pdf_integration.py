import numpy as np
import models.general_parameters as gp
import models.sigma0_xsections as sig0
#import fortran_integral
#import fortran_quad
# class vectorize(np.vectorize):
#     def __get__(self, obj, objtype):
#         return functools.partial(__call__, obj)
erroabs = 1
errorel = 1

#sighat = sighat
#p = pdf

    ## INTEGRATION PARAMETERS ##
    ## x_1 integration using LHAPDF6 for a DY like process (Field, R. D, Pertubative QCD, p. 174) ##

import lhapdf

@np.vectorize    
def dsigdM2_ISR(pdf, s, tau, Mmed, Q2mod, mx, gr, gl, gx, sighat):   ### Returns the cross section value in function of one or two of the parameters
    #Limits of integration
    Q2 = (Mmed*1000)**2  * Q2mod**2   ## Scale Q in terms of Mmed 
    a = tau
    b = 1
    p = pdf
    #Integration for all quarks flavors (reason for the loop)
    result_f = 0
    for x in gp.qid[1:4]:   ##NUMBER OF QUARKS IN THE SUM
        @np.vectorize
        def int_x1(x1):  ## integrand with PDFs
            return (p.xfxQ(gp.qid[x], x1, Q2)*p.xfxQ(-gp.qid[x], tau/x1, Q2 ) + (p.xfxQ(gp.qid[x], tau/x1, Q2)*p.xfxQ(-gp.qid[x], x1, Q2))  )/((tau/x1)*x1) \
                    * sighat/(x1)            
        ## Integration of the function above
        result_f += fortran_quad.quad16(int_x1, a, b)*(tau/Mmed)
        #result_f += (quad(int_x1, a, b, args=(s, 0, Mmed, mx, gr, gl, gx), limit=intlimit, epsabs=erroabs, epsrel=errorel)[0])*(tau/Mmed) ## x1 integration!! - The result is tau*\integral{f(x)} (mq is set to 0 because python errors in this phase)   
        #result_f = abs(result_f)  
    return result_f   #returns   dsig/dM^2 *  


# @np.vectorize    
# def dsigdM2(pdf, s, tau, Mmed, Q2mod, mx, gr, gl, gx, sighat):   ### Returns the cross section value in function of one or two of the parameters
#     #Limits of integration
#     Q2 = (Mmed**(1/2)*1000)  * Q2mod**2   ## Scale Q in terms of Mmed 
#     a = tau
#     b = 1
#     p = pdf
#     #Integration for all quarks flavors (reason for the loop)
#     result_f = 0
#     for x in qid[1:4]:   ##NUMBER O QUARKS IN THE SUM
#         @np.vectorize
#         def int_x1(xg, x1, s, mq, Mmed, mx, gr, gl, gx):  ## integrand with PDFs
#             return (p.xfxQ(qid[x], x1, Q2)*p.xfxQ(-qid[x], tau/x1, Q2 ) + (p.xfxQ(qid[x], tau/x1, Q2)*p.xfxQ(-qid[x], x1, Q2))  )/((tau/x1)*x1) \
#                     * sighat(xg, tau*s, mvec[x], Mmed, mx, gr, gl, gx)/(x1)            
#         ## Integration of the function above
#         result_f += dbintegrate_simp(int_x1, a, b, 6, s, 0, Mmed, mx, gr, gl, gx)*(tau/Mmed) ## x1 integration!! - The result is tau*\integral{f(x)} (mq is set to 0 because python errors in this phase)   
#         #result_f = abs(result_f)  
#     return result_f   #returns   dsig/dM^2 *




# @np.vectorize    
# def dsigdM2_noISR(pdf, s, tau, Mmed, Q2mod, mx, gr, gl, gx, sighat):   ### Returns the cross section value in function of one or two of the parameters
#     #Limits of integration
#     Q2 = (Mmed*1000)  * Q2mod**2   ## Scale Q in terms of Mmed 
#     a = tau
#     b = 1
#     p = pdf
#     #Integration for all quarks flavors (reason for the loop)
#     result_f = 0
#     for x in qid[1:4]:   ##NUMBER O QUARKS IN THE SUM
#         @np.vectorize
#         def int_x1(x1, xg, s, mq, Mmed, mx, gr, gl, gx):  ## integrand with PDFs
#             return (p.xfxQ(qid[x], x1, Q2)*p.xfxQ(-qid[x], tau/x1, Q2 ) + (p.xfxQ(qid[x], tau/x1, Q2)*p.xfxQ(-qid[x], x1, Q2))  )/((tau/x1)*x1) \
#                     * sighat/(x1)            
#         ## Integration of the function above
#         result_f += (quad(int_x1, a, b, args=(0, s, 0, Mmed, mx, gr, gl, gx), limit=intlimit, epsabs=erroabs, epsrel=errorel)[0])*(tau/Mmed) ## x1 integration!! - The result is tau*\integral{f(x)} (mq is set to 0 because python errors in this phase)   
#         #result_f = abs(result_f)  
#     return result_f   #returns   dsig/dM^2 *    

 

# @np.vectorize    
# def dsigdM2_ISR(pdf, s, tau, Mmed, Q2mod, mx, gr, gl, gx, sighat):   ### Returns the cross section value in function of one or two of the parameters
#     #Limits of integration
#     Q2 = (Mmed*1000)**2  * Q2mod**2   ## Scale Q in terms of Mmed 
#     a = tau
#     b = 1
#     p = pdf
#     #Integration for all quarks flavors (reason for the loop)
#     result_f = 0
#     for x in qid[1:4]:   ##NUMBER OF QUARKS IN THE SUM
#         @np.vectorize
#         def int_x1(x1):  ## integrand with PDFs
#             return (p.xfxQ(qid[x], x1, Q2)*p.xfxQ(-qid[x], tau/x1, Q2 ) + (p.xfxQ(qid[x], tau/x1, Q2)*p.xfxQ(-qid[x], x1, Q2))  )/((tau/x1)*x1) \
#                     * sighat/(x1)            
#         ## Integration of the function above
#         result_f += fortran_quad.quad16(int_x1, a, b)*(tau/Mmed)
#         #result_f += (quad(int_x1, a, b, args=(s, 0, Mmed, mx, gr, gl, gx), limit=intlimit, epsabs=erroabs, epsrel=errorel)[0])*(tau/Mmed) ## x1 integration!! - The result is tau*\integral{f(x)} (mq is set to 0 because python errors in this phase)   
#         #result_f = abs(result_f)  
#     return result_f   #returns   dsig/dM^2 *    





# @np.vectorize    
# def dsigdM2_noISR(pdf, s, tau, Mmed, Q2mod, mx, gr, gl, gx, sighat):   ### Returns the cross section value in function of one or two of the parameters
#     #Limits of integration
#     Q2 = (Mmed*1000)**2  * Q2mod**2   ## Scale Q in terms of Mmed 
#     a = tau
#     b = 1
#     p = pdf
#     #Integration for all quarks flavors (reason for the loop)
#     result_f = 0
#     for x in qid[1:4]:   ##NUMBER OF QUARKS IN THE SUM
#         @np.vectorize
#         def int_x1(x1):  ## integrand with PDFs
#             return (p.xfxQ(qid[x], x1, Q2)*p.xfxQ(-qid[x], tau/x1, Q2 ) + (p.xfxQ(qid[x], tau/x1, Q2)*p.xfxQ(-qid[x], x1, Q2))  )/((tau/x1)*x1) \
#                     * sighat/(x1)            
#         ## Integration of the function above
#         result_f += fortran_quad.quad16(int_x1, a, b)*(tau/Mmed)
#         #result_f += (quad(int_x1, a, b, args=(s, 0, Mmed, mx, gr, gl, gx), limit=intlimit, epsabs=erroabs, epsrel=errorel)[0])*(tau/Mmed) ## x1 integration!! - The result is tau*\integral{f(x)} (mq is set to 0 because python errors in this phase)   
#         #result_f = abs(result_f)  
#     return result_f   #returns   dsig/dM^2 *  



# @np.vectorize    
# def dsigdM2_noISR2(pdf, s, tau, Mmed, Q2mod, mx, gr, gl, gx, sighat):   ### Returns the cross section value in function of one or two of the parameters
#     #Limits of integration
#     Q2 = (Mmed*1000)  * Q2mod**2   ## Scale Q in terms of Mmed 
#     a = tau
#     b = 1
#     p = pdf
#     #Integration for all quarks flavors (reason for the loop)
#     result_f = 0
#     for x in qid[1:4]:   ##NUMBER O QUARKS IN THE SUM
#         @np.vectorize
#         def int_x1(x1, xg, s, mq, Mmed, mx, gr, gl, gx):  ## integrand with PDFs
#             return (p.xfxQ(qid[x], x1, Q2)*p.xfxQ(-qid[x], tau/x1, Q2 ) + (p.xfxQ(qid[x], tau/x1, Q2)*p.xfxQ(-qid[x], x1, Q2))  )/((tau/x1)*x1) \
#                     * sighat(xg, tau*s, mvec[x], Mmed, mx, gr, gl, gx)/(x1)            
#         ## Integration of the function above
#         result_f += (quad(int_x1, a, b, args=(0, s, 0, Mmed, mx, gr, gl, gx), limit=intlimit, epsabs=erroabs, epsrel=errorel)[0])*(tau/Mmed) ## x1 integration!! - The result is tau*\integral{f(x)} (mq is set to 0 because python errors in this phase)   
#         #result_f = abs(result_f)  
#     return result_f   #returns   dsig/dM^2 *  











# @np.vectorize
# def intxgamma(xg, s, mq, Mmed, mx, gr, gl, gx, sig0_factor):   
#     #Limits of integration
#     a = 0.010
#     b = 0.100
#     N = 20
#     #result_ee = complex_quadrature(integrand, a, b, args=(s, me, Mmed, mx, gr, gl, gx), limit=intlimit, epsabs=erroabs, epsrel=errorel)[0] ## x1 integration!! - The result is tau*\integral{f(x)} (mq is set to 0 because python errors in this phase)      
#     result_ee = quad(sig0_factor, a, b, args=(s, me, Mmed, mx, gr, gl, gx), limit=intlimit, epsabs=erroabs, epsrel=errorel)[0] ## x1 integration!! - The result is tau*\integral{f(x)} (mq is set to 0 because python errors in this phase)      
#     return result_ee   #returns   sig_tot for e+e- -> \gamma XX * 
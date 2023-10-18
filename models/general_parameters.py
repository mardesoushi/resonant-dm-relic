import numpy as np
### General Parameters


## SM PARAMETERS ## 
# MASSES (PDG 2018) (TeV) 
me = (511E-9)     #electron square mass  (TeV)
md = (2.2E-6) #  2.2 MeV +- 0.5 
mu = (4.7E-6) #  4.7 MeV +- 0.5
ms = (95E-6)  #  95 MeV +9 -3   
#mc = ...        #  1.275 GeV  +0.025 −0.035  
#mb = ...        #  4.18 GeV +0.04  −0.03 
#mt = ...        #  173.0 Gev +-0.04 
mmu = 105.66e-6
MZ = 91.1876e-3    # Z mass in TeV
MW = 80.379e-3     # W mass i TeV

mvec = [me, md, mu, ms, ms, mu, md] ## Quark masses vector
qid = [0, 1, 2, 3, -3, -2, -1]      ## quarks id (PDG) vector


#SM COUPLINGS
ge = 0.30282212088  #*np.pi*4                  #  ge = e (for h = c = 1 ref. Grif)  // alphaQED = e^2 / 4*Pi 
alph =  ge**2 / (4*np.pi)      # (1/137)    # QED coupling constant

thetaW =  28.759 * (np.pi/180)  ## Weinberg angle
cv = (-1/2) + (2 * (np.sin(thetaW)**2))
ca = -1/2
gz = ge / (np.sin(thetaW) * np.cos(thetaW))


## conversion from TeV^-2 to fb
brn = 0.3894*10**6       
#Nfermions = 6 + (6*3)   ## 24, e, mu, tau, ne, nmu, ntau,( u, d, c, s , t , b) * 3 colors

############################
## DARK MATTER PARAMETERS ## 
smax = 3.0**2 
Mmed = 3.0 # TeV
mx  = (Mmed/2)*0.80 # DM mass [TeV]

Nf_ee = 6
Nf_qq = 18


## Gauge Couplings DM ##
### Following V2 scenario ## gl = 0.01, gq = 0.1
#gr0 = 0.01             #right coupling constant with SM, according to CMS parameters
#gl0 = 0.01             ## vetorial: gl = gr  ## axial: gl = -gr ## chiral: gl = 0

gr0 = 0.01             #right coupling constant with SM, according to CMS parameters
gl0 = 0.01             ## vetorial: gl = gr  ## axial: gl = -gr ## chiral: gl = 0

gr0_q = 0.1             #right coupling constant with SM, according to CMS parameters
gl0_q = 0.1             ## vetorial: gl = gr  ## axial: gl = -gr ## chiral: gl = 0

gx0 = 1.   # dark coupling constant
grx0 = gxs0 = gxv0 = gx0  # specific SFV dark couplings
glx0 = 1.


### Auxiliar functions 
import pickle

# Salvar um objeto em formato pickle no disco
def save_obj(objeto, nome_arquivo):
    with open(nome_arquivo, 'wb') as arquivo:
        pickle.dump(objeto, arquivo)
    print(f"Objeto salvo em {nome_arquivo}")

# Carregar um objeto em formato pickle do disco
def load_obj(nome_arquivo):
    with open(nome_arquivo, 'rb') as arquivo:
        objeto = pickle.load(arquivo)
    return objeto
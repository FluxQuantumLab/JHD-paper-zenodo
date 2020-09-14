import numpy as np
import scipy.sparse as spa
import topo_generic as topo


def getH(param):
    """ Returns hamiltonian of cpp circuit, with convention taken from figure 2.a of https://arxiv.org/abs/2009.03291
    H is computed in the charge basis, in sparse format.
    Paramters:
        N: number of charge states. Need to be an odd integer. (N-1)/2 states positive and negative states are used, centered around zero.
        asym: each Ej_i and Cj_i is multiplied by asym[i]. This conserves the plasma frequency, and corresponds to changing the surface of the junctions.
    """    
    [N,Ec,Ej,ng1,ng2,phiX,asym]=param
    if N%2==0: 
        print(f'N={N} should be an odd integer')
        return

    cp=spa.diags(np.ones(N-1),1)
    cm=spa.diags(np.ones(N-1),-1)
    idn=spa.eye(N)

    # Josephson part of H
    V1=asym[0]*spa.kron(cp+cm,idn)
    V2=asym[1]*spa.kron(idn,cp+cm)
    V3=asym[2]*( np.exp(1j*phiX)*spa.kron(cp,cm) + np.exp(-1j*phiX)*spa.kron(cm,cp) )

    nd1=spa.diags(np.linspace((-N+1)/2-ng1,(N-1)/2-ng1,N),0)
    nd2=spa.diags(np.linspace((-N+1)/2-ng2,(N-1)/2-ng2,N),0)

    #Charging part of H
    C=[ [asym[0]+asym[2], -asym[2]], [-asym[2], asym[1]+asym[2]] ]
    Cinv=np.linalg.inv(C)
    
    T1=Cinv[0,0]*spa.kron(nd1**2,idn)
    T2=(Cinv[0,1]+Cinv[1,0])*spa.kron(nd1,nd2)
    T3=Cinv[1,1]*spa.kron(idn,nd2**2)

    H=-0.5*Ej*(V1+V2+V3) + Ec*(T1+T2+T3)
    return H


def minimize(x,*args):
    """ Returns excitation energy between ground state and first excited state. Use this for minimization routine SHGO.
    Parameters:
        - x are the optimization parameters, which are changed by SHGO routine
        - *args are the other fixed paramters
    """
    phiX,ng1,ng2=x
    [N,Ej,Ec,asym]=args
    param=[N,Ej,Ec,ng1,ng2,phiX,asym]

    E,V=topo.compute_ev(getH(param), nbands=2)
    return E[1]-E[0]

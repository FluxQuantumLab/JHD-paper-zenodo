import numpy as np
import scipy.sparse as spa
import topo_generic as topo

def getH(param):
    """ Returns hamiltonian of JHD circuit, with convention taken from figure 6 of https://arxiv.org/abs/2009.03291
    H is computed in the charge basis, in sparse format.
    Paramters:
        N: number of charge states. Need to be an odd integer. (N-1)/2 states positive and negative states are used, centered around zero.
        asym: each Ej_i and Cj_i is multiplied by asym[i]. This conserves the plasma frequency, and corresponds to changing the surface of the junctions.
    """

    [N,Ej,Ec,ng1,ng2,phiL,phiR,phiB,asym]=param
    if N%2==0: 
        print(f'N={N} should be an odd integer')
        return

    cp=spa.diags(np.ones(N-1),1)
    cm=spa.diags(np.ones(N-1),-1)
    idn=spa.eye(N)

    # Josephson part of H
    V1=asym[0]*spa.kron(cp+cm,idn)
    V2=asym[1]*spa.kron(idn,cp+cm)
    V3=asym[2]*(np.exp(1j*phiB)*spa.kron(cp,cm) + np.exp(-1j*phiB)*spa.kron(cm,cp))
    V4=asym[3]*(np.exp(1j*phiL)*spa.kron(cm,idn) + np.exp(-1j*phiL)*spa.kron(cp,idn))
    V5=asym[4]*(np.exp(1j*phiR)*spa.kron(idn,cp) + np.exp(-1j*phiR)*spa.kron(idn,cm))

    # capacitance matrix
    C=[ [asym[0]+asym[2]+asym[3],-asym[2]], [-asym[2],asym[1]+asym[2]+asym[4]] ]
    Cinv=np.linalg.inv(C)

    # Charging part of H
    nd1=spa.diags(np.linspace((-N+1)/2-ng1,(N-1)/2-ng1,N),0)
    nd2=spa.diags(np.linspace((-N+1)/2-ng2,(N-1)/2-ng2,N),0)
    T11=Cinv[0,0]*spa.kron(nd1**2,idn)
    T22=Cinv[1,1]*spa.kron(idn,nd2**2)
    T12=(Cinv[0,1]+Cinv[1,0])*spa.kron(nd1,nd2)

    H=-0.5*Ej*(V1+V2+V3+V4+V5)+Ec*(T11+T22+T12)
    return H

def getdH_LR(param):
    """ Returns the gradient of H wrto phi_L and phi_R only. Use this to compute Chern number associated to a phi_L phi_R plane.
    """
    [N,Ej,phiL,phiR,phiB,asym]=param
    cp=spa.diags(np.ones(N-1),1)
    cm=spa.diags(np.ones(N-1),-1)
    idn=spa.eye(N)

    dphiL=-asym[3]*(Ej/(2*1j))*( np.exp(-1j*phiL)*spa.kron(cp,idn) - np.exp(1j*phiL)*spa.kron(cm,idn) )
    dphiR=+asym[4]*(Ej/(2*1j))*( np.exp(1j*phiR)*spa.kron(idn,cp) - np.exp(-1j*phiR)*spa.kron(idn,cm) )

    return [dphiL, dphiR]


def getdH_full(param):
    """ Returns the full gradient of H
    """
    [N,Ej,Ec,ng1,ng2,phiL,phiR,phiB,asym]=param
    cp=spa.diags(np.ones(N-1),1)
    cm=spa.diags(np.ones(N-1),-1)
    idn=spa.eye(N)

    dphiL=-asym[3]*(Ej/(2*1j))*( np.exp(-1j*phiL)*spa.kron(cp,idn) - np.exp(1j*phiL)*spa.kron(cm,idn) )
    dphiR=+asym[4]*(Ej/(2*1j))*( np.exp(1j*phiR)*spa.kron(idn,cp) - np.exp(-1j*phiR)*spa.kron(idn,cm) )
    dphiB=-asym[2]*(Ej/(2*1j))*( np.exp(-1j*phiB)*spa.kron(cm,cp) - np.exp(1j*phiB)*spa.kron(cp,cm) )

    C=[ [asym[0]+asym[2]+asym[3],-asym[2]], [-asym[2],asym[1]+asym[2]+asym[4]] ]
    Cinv=np.linalg.inv(C)
    nd1=spa.diags(np.linspace((-N+1)/2-ng1,(N-1)/2-ng1,N),0)
    nd2=spa.diags(np.linspace((-N+1)/2-ng2,(N-1)/2-ng2,N),0)

    dng1=-2*Ec*(Cinv[0,0]*spa.kron(nd1,idn)+Cinv[0,1]*spa.kron(idn,nd2))
    dng2=-2*Ec*(Cinv[1,1]*spa.kron(idn,nd2)+Cinv[0,1]*spa.kron(nd1,idn))

    return [dphiL, dphiR, dphiB, dng1, dng2]

def minimize_phiL_phiR_ng(x,*args):
    """ Returns excitation energy between ground state and first excited state. Use this for minimization routine SHGO.
    Parameters:
        - x are the optimization parameters, which are changed by SHGO routine
        - *args are the other fixed paramters
    """
    phiL,phiR,ng=x
    ng1=ng2=ng
    #reshape (1,N) into (N) for shgo algorithm
    args=np.array(args).reshape(np.shape(args)[-1]) 
    [N,Ej,Ec,phiB,asym,tol]=args
    param=[N,Ej,Ec,ng1,ng2,phiL,phiR,phiB,asym]

    E,V=topo.compute_ev(getH(param), nbands=2, tol=tol)
    return E[1]-E[0]
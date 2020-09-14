import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

def braket(bra, ket):
    """returns braket of two states
    """
    return (bra.transpose().conjugate()).dot(ket)

def compute_ev(H, nbands=2, tol=1e-8):
    """ return first nbands sorted eigenvalues and eigenvectors, using Lanczos algorithm with numerical tolerance tol
    First state is ground state.
    """
    eige,eigv=eigsh(H,nbands,which='SA',return_eigenvectors=True,tol=tol)
    eigv=np.array(eigv)

    # Manually sorting eigenvalues and eigenvectors
    order=np.argsort(eige)
    eige_order=np.zeros(len(eige))
    eigv_order=[]
    for i in range(len(eige)):
        eige_order[i]=eige[order[i]]
        eigv_order.append(eigv[:,order[i]])
    eige=eige_order
    eigv=np.array(eigv_order)

    return eige,eigv


def compute_BC(H,dH,nbands=3,tol=1e-8):
    """ returns Berry curvature of H, summing over nbands states 
    Parameters:
        - H is the sparse matrix representation of the hamiltonian in the charge basis
        - dH is the sparse matrix representation of the gradient of the hamiltonian in the charge basis.
          This script works for 2D or 3D parameter space.
    Output:
        - Berry curvature vector. One element if dH is of size 2, three elements if dH is of size 3.
    """
    eige,eigv=eigsh(H,nbands,which='SA',return_eigenvectors=True,tol=tol)

    # Manually sorting eigenvalues and eigenvectors
    order=np.argsort(eige)
    eige_order=np.zeros(len(eige))
    eigv_order=[]
    for i in range(len(eige)):
        eige_order[i]=eige[order[i]]
        eigv_order.append(eigv[:,order[i]])
    eige=eige_order
    eigv=np.array(eigv_order)

    if len(dH)==2:
        BC=1j*0
        for i in range(1,nbands): #sum over excited states (excluding ground state)
            BC+=braket(eigv[0],dH[0]*eigv[i])*braket(eigv[i],dH[1]*eigv[0])/((eige[i]-eige[0])**2)
    elif len(dH)==3:
        BC=1j*np.zeros(3)
        for i in range(1,nbands): #sum over excited states (excluding ground state)
            for j in range(len(dH)): #repeat for each component of berry curvature
                BC[j]+=braket(eigv[0],dH[(j+1)%3]*eigv[i])*braket(eigv[i],dH[(j+2)%3]*eigv[0])/((eige[i]-eige[0])**2)
    else: print('dH has wrong size, need 2D or 3D gradient')
    return -2*np.imag(BC)
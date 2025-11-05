from numpy import zeros, block, identity
from modlatred import I
from cyclotomics import Cyclotomic
from math import sqrt
import numpy as np

def construct_BaiGalbraith_basis(K, r, m, q, A, b, t):
    """
    Given a Module-LWE instance (assuming short secret and error)
    Construct the Bai-Galbraith basis
     (qI_m   -A^T    b)
     (        I_r     )
     (               t)
    or more precisely, since we are working in column form, its transpose.
    """
    M = zeros((K.deg, m+r+1, m+r+1), dtype=int)
    # This 3d array represents a matrix over K.
    # The the (i,j)th element is a list of coefficients
    # given by M[:,i,j]

    M[0,:m,:m] = q*I(m)
    M[0, m:, m:] = I(r+1)
    M[0, -1, -1] = t

    # For each element of each sample, insert into M
    for i in range(m):
        for j in range(r):
            M[:,m+j,i] = -A[i,j,:]

    # For each element of b, insert into M
    for i in range(m):
        M[:,-1,i] = b[i]
    
    return M

def construct_l_embedding(K: Cyclotomic, r, m, q, A, b, sigma, l=1):
    """
    Given a Module-LWE instance (assuming short secret and error)
    Construct the (transpose) of the l-embedding basis
     (     B      |  b_1 , b_2, ... b_l)
     (____________|  _________________ )
     (            |          T_l       )
    where 
    - B is the cyclic embedding of the module q-ary lattice implied by the LWE instance,
    - b_1, ... b_l are the rotations by roots of unity of the vector b (in the cyclic embedding)
    - T_l = sigma sqrt(l) * identity(l) is the 'embedding submatrix'
    Note when l=1 this is just the unstructured Bai-Galbraith embedding with t=sigma.
    """
    d = K.deg
    c = K.cond
    assert 1 <= l <= d
    
    # Start with the structured embedding
    top_coeff = construct_BaiGalbraith_basis(K, r, m, q, A, b, t = sigma)
    top = block([[K.vOK_Zbasis(K.cyclic_embedding(top_coeff[:,i,j])) for j in range(m+r+1)] for i in range(m+r+1)])   

    # Remove the last (d-l) vectors and c coordinates
    if d-l > 0:
        top = top[:-(d-l),:]
    top = top[:,:-c]

    # Form the bottom matrix; note multiplication by sqrt(c) due to cyclic embedding
    t_coeff = round(sigma*sqrt(c*l))
    bot = block([zeros((l, d*m + d*r)), t_coeff * identity(l)]).T

    l_embedding = block([top, bot]).astype(dtype=int)

    return l_embedding
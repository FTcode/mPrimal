#!/usr/bin/env python3
import sys
sys.path.insert(1, './g6k')

from cyclotomics import Cyclotomic
from mPrimal import construct_BaiGalbraith_basis, construct_l_embedding
from primal_MLWE import generate_sspMLWE_instance
from numpy import block, array
from modlatred import ModuleLatticeReduction
import fpylll
from fpylll import IntegerMatrix, LLL
from fpylll.fplll.bkz_param import BKZParam
from fpylll.algorithms.bkz2 import BKZReduction
import matplotlib.pyplot as plt
from math import log
import numpy as np
from mBKZsim import MLWE_initial_Z_shape, mBKZsim_CN11
import time
from math import sqrt

def experiment_structured(K, r, m, q, A, s, e, b, sigma, betaQ, tours, ft = "ld"):
    # Note down the secret sublattice
    sigma_coeff = [0]*K.deg; sigma_coeff[0] = sigma
    V = block([K.vOK_Zbasis(K.cyclic_embedding(vi)) for vi in [*e, *s, sigma_coeff]])
    v = block([K.cyclic_embedding(vi) for vi in [*e, *s, sigma_coeff]])
    v_norm2 = (v @ v)

    # Initiate primal attack
    M = construct_BaiGalbraith_basis(K, r, m, q, A, b, t = sigma)
    R = block([[K.vOK_Zbasis(K.cyclic_embedding(M[:,i,j])) for j in range(m+r+1)] for i in range(m+r+1)])

    print("Initiating module lattice reduction.")
    mlr = ModuleLatticeReduction(R, K, float_type = ft)
    for run in range(5):
        print(f"LLL run {run}")
        mlr.lll(0,K.deg*(m+r+1),delta=.99)
        mlr.restructure()

    for tour in range(tours):
        print(f"Tour {tour+1}")
        tourstart = time.time()
        mlr.bkz(betaQ, 1)
        
        for run in range(5):
            mlr.lll(0,K.deg*(m+r+1),delta=.99)
            mlr.restructure()
            
        tourend = time.time()
        print(f"Time: {tourend-tourstart}s")
        print("Profile:", mlr.profile_K())  
        w = array(mlr.M.babai(v))        
        print("v in current basis:", w)

        # Succeed if any vector of length <=||v|| found anywhere in the basis
        for i in range(mlr.M.B.nrows):
            bi = array(mlr.M.B[i])
            if (bi @ bi) <= v_norm2:
                return True, tour+1

        # Succeed if first rank-1 module is vO_K
        succeed = True
        mlr.M.update_gso()
        for v in V:
            w = array(mlr.M.babai(v))
            if any(w[0+K.deg:]):
                succeed = False

        if succeed:
            return True, tour+1
    return False

def experiment_structured_progressive(K, r, m, q, A, s, e, b, sigma, tours, max_betaQ = 80, ft = "ld"):
    d = K.deg
    # Note down the secret sublattice
    sigma_coeff = [0]*K.deg; sigma_coeff[0] = sigma
    V = block([K.vOK_Zbasis(K.cyclic_embedding(vi)) for vi in [*e, *s, sigma_coeff]])
    v = block([K.cyclic_embedding(vi) for vi in [*e, *s, sigma_coeff]])
    v_norm2 = (v @ v)

    # Initiate primal attack
    M = construct_BaiGalbraith_basis(K, r, m, q, A, b, t = sigma)
    R = block([[K.vOK_Zbasis(K.cyclic_embedding(M[:,i,j])) for j in range(m+r+1)] for i in range(m+r+1)])
    Q_rank = K.deg*(m+r+1)

    print("Initiating module lattice reduction.")
    mlr = ModuleLatticeReduction(R, K, float_type = ft)
    for run in range(5):
        print(f"LLL run {run}")
        mlr.lll(0,Q_rank,delta=.99)
        mlr.restructure()

    for betaK in range(3, max_betaQ//d):
        print(f"betaK={betaK}")
        for tour in range(tours):
            print(f"tour={tour+1}")
            tourstart = time.time()
            mlr.bkz(beta = d*betaK, tours = 1)
            for run in range(5):
                mlr.lll(0,Q_rank,delta=.99)
                mlr.restructure()
            tourend = time.time()
            print(f"Time: {tourend-tourstart}s")

            # Succeed if any vector of length <=||v|| found anywhere in the basis
            for i in range(mlr.M.B.nrows):
                bi = array(mlr.M.B[i])
                if (bi @ bi) <= v_norm2:
                    return True, betaK*d, tour+1

            # Succeed if first rank-1 module is vO_K
            succeed = True
            mlr.M.update_gso()
            for v in V:
                w = array(mlr.M.babai(v))
                if any(w[0+K.deg:]):
                    succeed = False
            if succeed:
                return True, betaK*d, tour+1
    return False


def experiment_unstructured(K, r, m, q, A, s, e, b, sigma, betaQ, tours, l = 1, ft = "ld"):
    c = K.cond
    d = K.deg
    rank = d*(r+m) + l

    # Note down (one of) the secret vectors
    v = block([K.cyclic_embedding(vi) for vi in [*e, *s]]+[round(sigma*sqrt(c*l))]+[0]*(l-1))
    es_norm2 = (v[:-l] @ v[:-l])
    v_norm2 = (v @ v)
    
    # Generate the embedding matrix
    R = construct_l_embedding(K, r, m, q, A, b, sigma, l)

    # Run unstructured LLL and BKZ

    if use_fpylll:
        M = fpylll.GSO.Mat(IntegerMatrix.from_matrix(R), float_type = ft)
        bkz = BKZReduction(M)

        params_fplll = BKZParam(block_size = betaQ, strategies=fpylll.BKZ.DEFAULT_STRATEGY,
                                flags=0 | fpylll.BKZ.AUTO_ABORT | fpylll.BKZ.GH_BND | fpylll.BKZ.MAX_LOOPS,
                                max_loops=1)
    else:
        mlr = ModuleLatticeReduction(R, Cyclotomic(cond=1), float_type = ft)
        mlr.lll(0,rank,delta=.99)

    for tour in range(tours):
        print(f"Tour {tour+1}")
        tourstart = time.time()
        if use_fpylll:
            bkz.tour(params_fplll)
            LLL.reduction(M.B)
        else:
            mlr.bkz(betaQ, tours = 1)
        tourend = time.time()
        print(f"Time: {tourend-tourstart}s")

        # Succeed if error and secret found anywhere in the basis
        B = M.B if use_fpylll else mlr.M.B
        for i in range(B.nrows):
            bi = array(B[i])
            start = bi[:-l]
            if (start @ start) == es_norm2: 
                print("Found", bi)
                print(start)
                return True, tour+1
    return False

def experiment_unstructured_progressive(K, r, m, q, A, s, e, b, sigma, tours, l = 1, max_betaQ = 80, ft = "ld"):
    c = K.cond
    d = K.deg
    rank = d*(r+m) + l

    # Note down (one of) the secret vectors
    v = block([K.cyclic_embedding(vi) for vi in [*e, *s]]+[round(sigma*sqrt(c*l))]+[0]*(l-1))
    v_norm2 = (v @ v)
    
    # Generate the embedding matrix
    R = construct_l_embedding(K, r, m, q, A, b, sigma, l)

    # Run unstructured LLL and BKZ
    mlr = ModuleLatticeReduction(R, Cyclotomic(cond=1), float_type = ft)

    mlr.lll(0,rank,delta=.99)

    for betaQ in range(3, max_betaQ):
        print(f"betaQ={betaQ}")
        for tour in range(tours):
            print(f"tour={tour+1}")
            tourstart = time.time()
            mlr.bkz(betaQ, tours = 1)
            tourend = time.time()
            print(f"Time: {tourend-tourstart}s")

            # Succeed if a vector of length same as v is found anywhere in the basis
            for i in range(mlr.M.B.nrows):
                bi = array(mlr.M.B[i])
                if (bi @ bi) == v_norm2:
                    print("Found", bi)
                    return True, betaQ, tour+1
    return False

c = 3
q = 16000
r=30
m=30
sigma = 13
ft="dd"
betaQ = None
max_betaQ = None
tours = None
reject_sample_tol = None
structured = True
progressive = False
use_fpylll = False
l = 1

# Get command line arguments
for arg in sys.argv[1:]:
    exec(arg)

K = Cyclotomic(c)
d = K.deg

if tours is None:
    tours = 5

trials = 0
result = None

start_time = time.time()
while True:
    trials += 1
    A,s,e,b = generate_sspMLWE_instance(K,r,m,q,sigma,reject_sample_tol=reject_sample_tol)
    print(f"MLWE Experiment, c={c}, q={q}, r={r}, m={m}, sigma={sigma}")
    try:
        if structured:
            if progressive:
                print(f"Strategy: structured, progressive, tours={d*tours}, max_betaQ={max_betaQ}")
                result = experiment_structured_progressive(K, r, m, q, A, s, e, b, sigma, d*tours, max_betaQ, ft)
                break
            else:
                print(f"Strategy: structured, fixed-blocksize, betaQ={betaQ}, tours={d*tours}")
                result = experiment_structured(K, r, m, q, A, s, e, b, sigma, betaQ, d*tours, ft = ft)
                break
        else:
            if progressive:
                print(f"Strategy: unstructured (l={l}), progressive, tours={tours}, max_betaQ={max_betaQ}")
                result = experiment_unstructured_progressive(K, r, m, q, A, s, e, b, sigma, tours, l, max_betaQ, ft)
                break
            else:
                print(f"Strategy: unstructured (l={l}), fixed-blocksize, betaQ={betaQ}, tours={tours}")
                result = experiment_unstructured(K, r, m, q, A, s, e, b, sigma, betaQ, tours, ft = ft, l=l)
                break
    except Exception as err:
        if str(err) in ['']:#["b'infinite loop in babai'", "math domain error", "b'success'", "delta must be > 0.25"]:
            print("Error:", str(err))
            print("Generating new instance.")
            continue
        else:
            raise err
end_time = time.time()

print(f"minutes = {(end_time-start_time)/60}")
print(f"success = {result}")
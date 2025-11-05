from cyclotomics import Cyclotomic
from numpy import array, rint, ceil, pad, reshape, block, allclose, roll
from numpy.random import normal, randint
from sympy import cyclotomic_poly
from numpy.polynomial import polynomial
from math import sqrt
from scipy.stats import chi2
import matplotlib.pyplot as plt
import numpy as np

def module_coeff_inner_product(K, polys1, polys2, conjugate = False):
    """
    Computes K-inner product between elements of K^r
    where each elt is represented as a list of coeff-embeddings
    """
    assert len(polys1) == len(polys2)
    r = len(polys1)

    result = sum([K.ip_K_coeff(polys1[i], polys2[i], conjugate) for i in range(r)])
    phi_c = array(cyclotomic_poly(K.cond).as_poly().all_coeffs(), dtype="int")
    _, result = polynomial.polydiv(result, phi_c)
    # Pad zeroes
    if len(result) < K.deg:
        result = pad(result, (0, K.deg - len(result)))

    # Check against cyclic embedding
    p1c = array([K.cyclic_embedding(polys1[i]) for i in range(r)]).flatten()
    if not conjugate:
        polys2 = [K.coeff_conjugate(p) for p in polys2]
    p2c = array([K.cyclic_embedding(polys2[i]) for i in range(r)]).flatten()
    cyclic_check = K.ip_K(p1c, p2c)
    assert all(K.cyclic_embedding(result) == cyclic_check)

    return result

def generate_LWE_noise(K : Cyclotomic, sigma):
    """
    Implementation of the noise generation inspired by [DD12]
    Returns a polynomial in Z[X]/Phi_c(X) in coefficient embedding
    with expected squared Euclidean norm d*(sigma**2).
    """
    c = K.cond

    # Sample a Gaussian poly in Q[X]/(X^c - 1)
    # i.e. just c random coefficients
    coeffs = array(normal(loc=0, scale=sigma/sqrt(c), size=c), dtype="float")

    # Reduce modulo the cyclotomic polynomial Phi_c(X)
    phi_c = array(cyclotomic_poly(K.cond).as_poly().all_coeffs(), dtype="int")
    _, remainder = polynomial.polydiv(coeffs, phi_c)

    # Take remainder and round each coefficient
    noise = [0] * K.deg
    noise[:len(remainder)] = array(rint(remainder), dtype = "int")

    return noise

def generate_LWE_noise_vector(K, sigma, rank, reject_sample_tol = None):
    """
    Use generate_LWE_noise to generate a noise vector of given K-rank.
    If eps=reject_sample_tol is set, will reject until
        (1-eps)*sigma*sqrt(d*rank) <= norm <= (1+eps)*sigma*sqrt(d*rank)
    i.e., guaranteed Euclidean norm is close to expectation.  
    """

    condition = False
    eps = reject_sample_tol
    d = K.deg

    if not (eps is None):
        mean = sigma * sqrt(d * rank)
        lower = (1 - eps) * mean
        upper = (1 + eps) * mean
        #print(f"Mean {mean}")
        #print(f"Accepting range ({lower}, {upper})")

    trials = 0
    while not condition:
        trials += 1
        condition = False
        vec = array([generate_LWE_noise(K, sigma) for _ in range(rank)])
        cyclic = array([K.cyclic_embedding(vec[i]) for i in range(rank)]).flatten()

        norm = sqrt(K.ip_Q(cyclic, cyclic))

        condition = (eps is None) or (lower <= norm <= upper) 
        # short circuit evaluation of "or" means it doesnt matter if lower,upper undefined

        if condition and not (reject_sample_tol is None):
            print(f"Accepting norm {norm} after {trials} trials.")
    return vec

def generate_sspMLWE_instance(K, r, m, q, sigma, conjugate = False, reject_sample_tol = None):
    """
    Generate a short-secret 'primal'-module-LWE instance over K
        A is m random elements of (R_q)^r
        s is a random short element of (R_q)^r
        e is a random short element in (R_q)^m
        b has components b_i = <a_i, s>_K + e_i (mod q)
    - Conjugate controls whether the second term in IP is conjugated or not
    - reject_sample_tol controls norm of e and s
    """
    
    # A is m random element of (R_q)^r
    A = randint(0, q-1, size=(m, r, K.deg))

    # Secret is a random short element of (R_q)^r
    s = generate_LWE_noise_vector(K, sigma, r, reject_sample_tol)

    # Error is a random short element in (R_q)^m
    e = generate_LWE_noise_vector(K, sigma, m, reject_sample_tol)
    
    # Compute b = As + e
    # ith component is <a_i, s_i> + e_i (mod q)
    b = array([(module_coeff_inner_product(K, A[i], s, conjugate) + e[i])%q for i in range(m)], dtype = "int")
    
    return (A, s, e, b)

def DD12_squarednorm_pmf(Q_rank, sigma, shift, tol = 0.001):
    pmf = dict()

    # Parameters for chi-squared distribution
    df = Q_rank
    scale = sigma**2
    loc = shift

    # Compute tails: only sum over region which contains (1 - 2 * tol) of probability
    lower_tail = int(chi2.ppf(q = tol, df = df, scale = scale, loc = loc))
    upper_tail = int(chi2.isf(q = tol, df = df, scale = scale, loc = loc))

    for v_norm2 in range(lower_tail, upper_tail + 1):
        if v_norm2 == lower_tail:
            pmf[v_norm2] = chi2.cdf(v_norm2 + 1/2, df=df, scale=scale, loc=loc)
        elif v_norm2 == upper_tail:
            pmf[v_norm2] = 1 - chi2.cdf(upper_tail - 1/2, df=df, scale=scale, loc=loc)
        else:
            pmf[v_norm2] = chi2.cdf(v_norm2 + 1/2, df=df, scale=scale, loc=loc) - chi2.cdf(v_norm2 - 1/2, df=df, scale=scale, loc=loc)

    assert allclose(sum(pmf.values()), 1)
    return pmf
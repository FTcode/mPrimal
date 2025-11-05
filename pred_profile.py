
from modlatred import ModuleLatticeReduction, random_qary_cyclotomic
from cyclotomics import Cyclotomic
from math import log
from numpy import array
from mBKZsim import mBKZsim_CN11, initial_Z_shape_PV21
from predictions import slope_K
import matplotlib.pyplot as plt
from read_csv import read_csv
import csv

"""
Write the mGSA, predicted profiles, and experimental profiles (from data/profiles_d160_beta64)
all to a single CSV file.
"""

for c in [1, 3, 4, 15, 16]:
    betaQ = 64
    tau = 30
    K = Cyclotomic(c)
    d = K.deg

    mQ = 160
    nQ = c * (mQ//(3*d))
    q = 16317
    
    profiles = {}

    # Get an initial profile
    K_profile = initial_Z_shape_PV21(K, q, mQ//d, nQ//d, sigma=1)
    ln_vol = sum(K_profile)
    
    # Normalise
    K_profile = K_profile - ln_vol / len(K_profile)

    assert betaQ % d == 0
    betaK = betaQ // d

    # GSA-predicted profile
    print("Calculating mGSA profile")
    mGSA_slope = slope_K(K, betaK)
    mGSA = array(range(mQ//d)) * (mGSA_slope)
    # Normalise
    mGSA = mGSA - sum(mGSA)/len(mGSA)
    mGSA /= d
    profiles['mGSA'] = mGSA

    # Simulator-predicted profiles
    print("Simulating profiles")
    sim = mBKZsim_CN11(K, K_profile, tau, betaK)
    # Normalise
    sim -= sum(sim)/len(sim)
    sim /= d
    profiles['pred'] = sim

    # Get real profiles
    filename = f'./data/profiles_d160_beta64/prof_c{c}_avg.csv'
    data = read_csv(filename)['ellK']
    data = list(filter(lambda x : not x is None, data))
    profiles['exp'] = data

    # Write all to csv
    csv_cols = [['i'] + list(range(mQ))]
    for key, value in profiles.items():
        csv_cols.append([key, *value])
    csv_rows = zip(*csv_cols)

    writer = csv.writer(open(f'./data/profiles_d160_beta64/sim_c{c}.csv', 'w+'))
    writer.writerows(csv_rows)

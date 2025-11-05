from modlatred import random_qary_cyclotomic, ModuleLatticeReduction
from cyclotomics import Cyclotomic
from numpy import zeros, ones
import matplotlib.pyplot as plt
import csv

def experimental_mHKZ_shape(K, rank, samples):
    """
    Returns an experimental mHKZ shape, averaged over a specific number of samples.
    """
    tot_prof = zeros(rank)
    for _ in range(samples):
        print(f"Cond {K.cond} rank {rank} sample {_}")
        while True:
            try:
                # Generate a random lattice
                B = random_qary_cyclotomic(K, rank, rank//2, q=16317)
                # Take the scaled HKZ profile
                mlr = ModuleLatticeReduction(B, K)
                mlr.hkz()
                break
            except:
                pass
        prof = mlr.profile_K(rescale = True)
        tot_prof += prof
    
    tot_prof /= samples
    tot_prof -= ones(rank) * (sum(tot_prof)/rank) #eliminate compounding rounding errors?
    return tot_prof

if __name__ == '__main__':

    cs = [1, 3, 4, 5, 7, 8, 9, 11, 13, 15, 16, 17]
    Q_heur_rank = 45 # Q-rank where heuristics kick in

    writer = csv.writer(open("data/exp_mHKZ_profiles.csv", 'a'))

    for c in cs:
        K = Cyclotomic(c)
        K_heur_rank = Q_heur_rank // K.deg
        prof = experimental_mHKZ_shape(K, K_heur_rank, 500)
        writer.writerow([c, *prof])

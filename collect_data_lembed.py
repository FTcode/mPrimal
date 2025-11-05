from mBKZsim import unsPrimal_success, lembedding_success
from cyclotomics import Cyclotomic
import csv

parent_dir = "./data/lembed/"
for dir in [parent_dir+"fpylll/", parent_dir+"g6k/"]:
    ls = range(1, 9)
    csv_rows = []
    csv_rows.append(["l", "exp", "pred", "pred_lower"])
    trials = 200

    c = 15
    K = Cyclotomic(c)
    r = 14
    m = 15
    sigma = 4
    q = 2100
    tours = 10
    betaQ = 58

    for l in ls:
        wins = 0
        for trial in range(1,trials+1):
            filename = dir + f"c15_l{l}_trial{trial}.txt"
            with open(filename, 'r') as file:
                lines = file.readlines()
            exec(lines[-1])
            if success: wins += 1
        exps = wins/trials
        pred = lembedding_success(K, r, m, q, sigma, betaQ, tours, l)
        pred_lower = unsPrimal_success(K, r, m, q, sigma, betaQ, tours, l)
        csv_rows.append([l, exps, pred, pred_lower])

    writer = csv.writer(open(dir+"lembed_all.csv", 'w'))
    writer.writerows(csv_rows)
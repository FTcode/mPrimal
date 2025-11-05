from mBKZsim import mPrimal_success, unsPrimal_success, mPrimal_success_progressive, unsPrimal_success_progressive
from cyclotomics import Cyclotomic
import csv

PARAM_SETS = [
	("c3", list(range(48,82,2)), 3, 59, 60, 11, 48000, 5),
	("c5", list(range(44,84,4)), 5, 29, 30, 12, 30000, 5),
	("c7", list(range(42,78,6)), 7, 18, 21, 10, 10000, 5),
	("c8", list(range(44,76,4)), 8, 29, 30, 8,  20000, 5),
]

trials = 100
prog_trials = 250

for c_str, bQs, c, r, m, sigma, q, tours in PARAM_SETS:
	print(f"==== {c_str} ====")
	dir = f"./data/mPrimal240/{c_str}/"
	csv_out = dir + "bkz_all.csv"
	csv_cols = []
	all_bQs = list(range(min(bQs), max(bQs) + 1))
	csv_cols.append(["bQ", *all_bQs])
	K = Cyclotomic(c)
	d = K.deg

	for struct in [True, False]:
		# Non-progressive
		col = [f"struct{struct}_exp"]
		for bQ in all_bQs:
			if not bQ in bQs:
				col.append("")
				continue
			wins = 0
			total = 0
			for trial in range(1, trials+1):
				filename = dir + f"betaQ{bQ}_struct{struct}_trial{trial}.txt"
				with open(filename, 'r') as file:
					lines = file.readlines()
				try:
					lastline = lines[-1]
					assert lastline.startswith("success = ")
					success = None
					exec(lastline)
					if success: wins += 1
					total += 1
				except Exception as err:
					print("Bad experiment,", filename)
			col.append("{:.6f}".format(wins/total))
		csv_cols.append(col)

		col = [f"struct{struct}_pred"]
		for bQ in all_bQs:
			if struct:
				if not bQ in bQs:
					col.append("")
					continue
				else:
					pred = mPrimal_success(K, r, m, q, sigma, bQ//d, tours*d)
			else:
				pred = unsPrimal_success(K, r, m, q, sigma, bQ, tours)
			col.append("{:.6f}".format(pred))
		csv_cols.append(col)

	for struct in [True,False]:
		if c != 3: continue

		# Progressive predictions
		if struct:
			pred = mPrimal_success_progressive(K, r, m, q, sigma, tours = 4, max_blocksize_K = 80//d)
		else:
			pred = unsPrimal_success_progressive(K, r, m, q, sigma, tours = 2, max_blocksize_Q = 80)

		# Progressive experiments
		wins = []
		exps = 0
		for trial in range(1, prog_trials+1):
			filename = dir + f"prog_struct{struct}_trial{trial}.txt"
			with open(filename,'r') as file:
				lines = file.readlines()
			try:
				lastline = lines[-1]
				assert lastline.startswith("success =")
				success = None
				exec(lastline)
				exps += 1
				if success: wins.append(success)
			except Exception as err:
				print("Bad experiment", filename)
		# Form the cmf
		col = [f"prog_struct{struct}_exp"]
		pred_col = [f"prog_struct{struct}_pred"]

		for bQ in all_bQs:
			if struct and not (bQ in bQs):
				col.append('')
				pred_col.append('')
				continue
			col.append("{:.6f}".format(len([*filter(lambda x : x[1] <= bQ, wins)])/prog_trials))
			beta = bQ//d if struct else bQ
			pred_col.append("{:.6f}".format(pred[beta] if beta in pred.keys() else 1))
		print(struct, col)
		print(struct, pred_col)
		csv_cols.append(col)
		csv_cols.append(pred_col)

	writer = csv.writer(open(csv_out, 'w'))
	writer.writerows(zip(*csv_cols))
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from data_modules import STRDataset


def find_patterns(seq, X, Y, max_d=3):
	# do poly adj seqs (case 2 and 4)
	X_d = 0
	Y_d = 0
	last_X = 0
	last_Y = 0
	X_poly_res = dict()
	Y_poly_res = dict()

	for i,base in enumerate(seq):
		if base != X:
			X_poly_res[X_d] = min(i, last_X)
			X_d += 1
			if X_d > max_d:
				break
		else:
			last_X = i + 1

	for i,base in enumerate(seq):
		if base != Y:
			Y_poly_res[Y_d] = min(i, last_Y)
			Y_d += 1
			if Y_d > max_d:
				break
		else:
			last_Y = i + 1

	# do wildcard cases
	X_d = 0
	Y_d = 0
	last_X = 0
	last_Y = 0
	X_wc_res = dict()
	Y_wc_res = dict()

	for i,base in enumerate(seq):
		if i % 2 == 0:
			continue
		elif base != X:
			X_wc_res[X_d] = min(i, last_X)
			X_d += 1
			if X_d > max_d:
				break
		else:
			last_X = i + 1
	
	for i,base in enumerate(seq):
		if i % 2 != 0:
			continue
		elif base != Y:
			Y_wc_res[Y_d] = min(i, last_Y)
			Y_d += 1
			if Y_d > max_d:
				break
		else:
			last_Y = i + 1

	# do with Z
	X_d = 0
	X_Z_d = 0
	X_is_Z_d = 0
	Y_d = 0
	Y_Z_d = 0
	Y_is_Z_d = 0
	last_X = 0
	last_Y = 0
	X_Z_res = dict()
	Y_Z_res = dict()
	X_Z_pos = dict()
	Y_Z_pos = dict()

	last_Z = 0

	for i,base in enumerate(seq):
		if i % 2 == 0:
			if base == X:
				if last_Z == 0 or last_X == 0:
					X_Z_res[X_d + X_Z_d + X_is_Z_d] = 0
				else:
					# print('A!', X_d + X_Z_d + X_is_Z_d)
					X_Z_res[X_d + X_Z_d + X_is_Z_d] = min(i, last_X, last_Z) + 1
				X_is_Z_d += 1
				if X_d + X_Z_d + X_is_Z_d > max_d:
					break
			else:
				prev_X_Z_d = X_Z_d
				if base in X_Z_pos.keys():
					X_Z_pos[base].append(i + 1)
				else:
					X_Z_pos[base] = [i + 1]

				# get distance from Z positions
				Z_counts = [len(v) for v in X_Z_pos.values()]
				X_Z_d = sum(Z_counts) - max(Z_counts)

				if X_Z_d > prev_X_Z_d:
					if last_Z == 0 or last_X == 0:
						X_Z_res[X_d + prev_X_Z_d + X_is_Z_d] = 0
					else:
						# print('B!', X_d + prev_X_Z_d + X_is_Z_d)
						X_Z_res[X_d + prev_X_Z_d + X_is_Z_d] = min(i, last_X, last_Z) + 1
					if X_d + X_Z_d + X_is_Z_d > max_d:
						break

				# get last Z position
				max_count = max(Z_counts)
				last_Z = min(
					[v[-1] for v in X_Z_pos.values() if len(v) == max_count]
				)
		elif base == X:
			last_X = i + 1
		else:
			if last_Z == 0 or last_X == 0:
				X_Z_res[X_d + X_Z_d + X_is_Z_d] = 0
			else:
				# print('C!', X_d + X_Z_d + X_is_Z_d)
				X_Z_res[X_d + X_Z_d + X_is_Z_d] = min(i, last_X, last_Z) + 1
			X_d += 1
			if X_d + X_Z_d + X_is_Z_d > max_d:
				break

	last_Z = 0

	for i,base in enumerate(seq):
		if i % 2 != 0:
			if base == Y:
				if last_Z == 0 or last_Y == 0:
					Y_Z_res[Y_d + Y_Z_d + Y_is_Z_d] = 0
				else:
					Y_Z_res[Y_d + Y_Z_d + Y_is_Z_d] = min(i, last_Y, last_Z) + 1
				Y_is_Z_d += 1
				if Y_d + Y_Z_d + Y_is_Z_d > max_d:
					break
			else:
				prev_Y_Z_d = Y_Z_d
				if base in Y_Z_pos.keys():
					Y_Z_pos[base].append(i + 1)
				else:
					Y_Z_pos[base] = [i + 1]

				# get distance from Z positions
				Z_counts = [len(v) for v in Y_Z_pos.values()]
				Y_Z_d = sum(Z_counts) - max(Z_counts)

				if Y_Z_d > prev_Y_Z_d:
					if last_Z == 0 or last_Y == 0:
						Y_Z_res[Y_d + prev_Y_Z_d + Y_is_Z_d] = 0
					else:
						Y_Z_res[Y_d + prev_Y_Z_d + Y_is_Z_d] = min(i, last_Y, last_Z) + 1
					if Y_d + Y_Z_d + Y_is_Z_d > max_d:
						break

				# get last Z position
				last_Z = min(
					[v[-1] for v in Y_Z_pos.values() if len(v) == max(Z_counts)]
				)
		elif base == Y:
			last_Y = i + 1
		else:
			if last_Z == 0 or last_Y == 0:
				Y_Z_res[Y_d + Y_Z_d + Y_is_Z_d] = 0
			else:
				Y_Z_res[Y_d + Y_Z_d + Y_is_Z_d] = min(i, last_Y, last_Z) + 1
			Y_d += 1
			if Y_d + Y_Z_d + Y_is_Z_d > max_d:
				break

	return {
		1: X_Z_res,
		2: X_poly_res,
		3: Y_Z_res,
		4: Y_poly_res,
		5: X_wc_res,
		6: Y_wc_res,
	}


if __name__ == '__main__':
	# options
	training_params = {
		# Data File
		'data_dir': os.path.join('..', 'data', 'heterozygosity'),
		'data_fname': 'sample_data_dinucleotide_mfr0_005_mnc2000.json',

		# Data Module
		'min_copy_number': None,
		'max_copy_number': 64,
		'incl_STR_feat': True,
		'min_boundary_STR_pos': 6,
		'max_boundary_STR_pos': 6,
		'window_size': 50,
		'bp_dist_units': 1000.0,
	}

	# Load with DataModule
	data_path = os.path.join(
		'..', '..', 'data', 'heterozygosity', training_params['data_fname']
	)
	data_module = STRDataset(
		data_path,
		split_name=None,
		incl_STR_feat=training_params['incl_STR_feat'],
		min_boundary_STR_pos=training_params['min_boundary_STR_pos'],
		max_boundary_STR_pos=training_params['max_boundary_STR_pos'],
		window_size=training_params['window_size'],
		min_copy_num=training_params['min_copy_number'],
		max_copy_num=training_params['max_copy_number'],
		bp_dist_units=training_params['bp_dist_units'],
		return_strings=True,
		return_data=True
	)
	samples = data_module.data

	# add 'sample_id' col so saved sample data can be joined with 
	#  saved found patterns
	samples['sample_id'] = samples.index.values
	#samp = samples.iloc[120007]

	# For all samples find patterns
	pattern_res = []

	for _, samp in tqdm(samples.iterrows(), total=len(samples)):
		pre_X = samp.str_seq[0]
		pre_Y = samp.str_seq[1]
		post_X = samp.str_seq[-1]
		post_Y = samp.str_seq[-2]

		# get feature counts
		pre_feats = find_patterns(np.array(list(reversed(samp.pre_seq))), pre_X, pre_Y)
		post_feats = find_patterns(np.array(list(samp.post_seq)), post_X, post_Y)

		for case, feat_vals in pre_feats.items():
			for d, v in feat_vals.items():
				pattern_res.append({
					'sample_id': samp.sample_id,
					'case': case,
					'd': d,
					'n_bases': v,
					'X': pre_X,
					'Y': pre_Y,
					'pre': True,
					'str_len': samp.motif_len * samp.num_copies,
				})

		for case, feat_vals in post_feats.items():
			for d, v in feat_vals.items():
				pattern_res.append({
					'sample_id': samp.sample_id,
					'case': case,
					'd': d,
					'n_bases': v,
					'X': post_X,
					'Y': post_Y,
					'pre': False,
					'str_len': samp.motif_len * samp.num_copies,
				})

	# Save results
	pattern_res_df = pd.DataFrame(pattern_res)

	task_version_dir = '{}-m{}'.format(
		training_params['data_fname'].split('.')[0].split('leotide_')[1],
		str(training_params['max_copy_number']).replace('.', '_')
	)
	task_version_dir = os.path.join('find_patterns_output', task_version_dir)
	if not os.path.exists(task_version_dir):
		os.makedirs(task_version_dir)

	pattern_res_df.to_csv(
		os.path.join(task_version_dir, 'pattern_res.csv'), 
		index=False
	)
	samples.to_csv(
		os.path.join(task_version_dir, 'samples.csv'), 
		index=False
	)
"""Creates input feature representation for labeled samples, finds
corresponding feature string and chromosome range, removes STRs with
length over threshold, and saves resulting samples in a JSON fine

Updates over preprocessV2_het_multi.py:
	- Samples from HipSTR_name will be in same train/val/test split. This
		is important for when using types of motifs who are complements of
		each other.
"""

import json
import os
import sys
import math

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def make_feat_mat(seq_string, distance):
	bases = np.array(list(seq_string))
	unk = (bases == 'N').astype(int) / 4

	A = unk + (bases == 'A')
	C = unk + (bases == 'C')
	G = unk + (bases == 'G')
	T = unk + (bases == 'T')

	return np.vstack((A, C, G, T, distance))


def is_perfect_repeat(seq, motif):
	# Check is motif matches forward
	if seq[:len(motif)] == motif:
		pass
	# If matches reversed, reverse motif
	elif seq[:len(motif)] == motif[::-1]:
		motif = motif[::-1]
	else:
		return False

	# Check that seq is just the motif repeated
	for i in range(len(seq)):
		if seq[i] != motif[i%len(motif)]:
			return False

	return True


def make_stratified_splits(group_label, train_ratio, test_ratio, rand_seed=None):
	""" Makes random train, val, and test splits stratified by some 
		group_label. Returns results as a vector where 0 is train,
		1 is val, and 2 is test.
	"""
	other_inds, test_inds = next(StratifiedShuffleSplit(
		test_size=test_ratio, 
		n_splits=1, 
		random_state=rand_seed).split(group_label, group_label))
	train_inds, val_inds = next(StratifiedShuffleSplit(
		test_size=(1-test_ratio-train_ratio)/(1-test_ratio),
		n_splits=1,
		random_state=rand_seed).split(group_label[other_inds], group_label[other_inds]))

	# Make return vector of split labels
	split_labels = np.zeros(len(group_label), dtype=int)
	split_labels[other_inds[val_inds]] = 1
	split_labels[test_inds] = 2

	return split_labels

if __name__ == '__main__':
	""" Final preprocessing step for STRs.

	Samples that don't meet the criteria to be labeled stable or unstable
	are excluded.

	Will then generate splits based on copy number, such that dataloader can 
	restrict the copy number range and have balanced groups down the road.

	Parameters:
		samp_dir: path to dir with labeled samples json
		samp_fname: name of json file with labeled samples
		save_dir: path to dir to save output preprocessed samples json
		this_sample_set_fname: name of output json file. If None, will be
			'sample_data_dinucleotide_t{stable_minor_freq_max}_{unstable_minor_freq_min}.json'
			where {stable_minor_freq_max} and {unstable_minor_freq_min} are 
			values after the decimal point for the parameters below.
		max_STR_len: maximum length of STR to keep in number of bps.
		min_num_called: minimum number of samples called to keep STR. If None,
			keep all STRs.
		motif_types: Motifs not in this list will be removed. Remember 
			labeled_samples already includes complements.
		stable_minor_freq_max: Samples with a heterozygosity score <= this 
			will be labeled as 0.
		unstable_minor_freq_min: Samples with a heterozygosity score > this 
			will be labeled as 1.
	"""
	samp_dir = os.path.join('..', 'data', 'heterozygosity')
	samp_fname = 'labeled_samples_dinucleotide.json'
	save_dir = os.path.join('..', 'data', 'heterozygosity')
	this_sample_set_fname = None

	max_STR_len = 50
	min_num_called = 2000
	motif_types = ['GT', 'TG', 'CT', 'TC', 'AC', 'CA', 'AT', 'TA', 'AG', 'GA']
	stable_minor_freq_max = 0.0
	unstable_minor_freq_min = 0.0025

	if this_sample_set_fname is None:
		this_sample_set_fname = 'sample_data_dinucleotide_mfr{:}_{:}_mnc{:}.json'.format(
			str(stable_minor_freq_max).split('.')[1],
			str(unstable_minor_freq_min).split('.')[1],
			min_num_called
		)

	# Load labeled STRs to be preprocessed
	samp_path = os.path.join(samp_dir, samp_fname)

	with open(samp_path) as fp:    
		samples = json.load(fp)

	filtered_sample_data = []
	labels = []

	# Filter samples, then save formatted data
	for i in tqdm(range(len(samples)), file=sys.stdout):
		samp_dict = samples.pop(0)

		# filter out by STR motif type
		if samp_dict['motif'] not in motif_types:
			continue

		# filter out by STR length
		if samp_dict['motif_len'] * samp_dict['num_copies'] > max_STR_len:
			continue

		# filter out by min num called
		if min_num_called is not None and samp_dict['num_called'] < min_num_called:
			continue

		# Determine label or if sample should be excluded
		is_stable = samp_dict['minor_freq'] <= stable_minor_freq_max
		is_unstable = samp_dict['minor_freq'] > unstable_minor_freq_min
		assert not (is_stable and is_unstable)

		if is_stable:
			samp_dict['label'] = 0
		elif is_unstable:
			samp_dict['label'] = 1
		else:
			continue
		filtered_sample_data.append(samp_dict)

		# # for dev
		# if i > 10000:
		# 	break

	# Complement sequences must also be in the same folds.
	samp_df = pd.DataFrame(filtered_sample_data)
	hipstr_labels = samp_df.groupby('HipSTR_name', as_index=False).num_copies.mean()

	# make spits as HipSTR_name level
	hipstr_labels['split_1'] = make_stratified_splits(
		hipstr_labels['num_copies'], 0.7, 0.15, 36
	)
	hipstr_labels['split_2'] = make_stratified_splits(
		hipstr_labels['num_copies'], 0.7, 0.15, 147
	)
	hipstr_labels['split_3'] = make_stratified_splits(
		hipstr_labels['num_copies'], 0.7, 0.15, 12151997
	)

	# Map to full sample dataframe
	hipstr_label_map = hipstr_labels.set_index('HipSTR_name').to_dict('index')
	samp_df['split_1'] = samp_df.HipSTR_name.apply(
		lambda x: hipstr_label_map[x]['split_1']
	)
	samp_df['split_2'] = samp_df.HipSTR_name.apply(
		lambda x: hipstr_label_map[x]['split_2']
	)
	samp_df['split_3'] = samp_df.HipSTR_name.apply(
		lambda x: hipstr_label_map[x]['split_3']
	)

	# Plot copy number distribution by split to verify that splits are balanced
	# sns.countplot(x='num_copies', hue='split_1', data=samp_df)
	# plt.show()
	# sns.catplot(x='num_copies', hue='split_1', row='label', data=samp_df,
	# 	kind='count', aspect=2)
	# plt.show()
	# sns.catplot(x='num_copies', hue='split_2', row='label', data=samp_df,
	# 	kind='count', aspect=2)
	# plt.show()
	# sns.catplot(x='num_copies', hue='split_3', row='label', data=samp_df,
	# 	kind='count', aspect=2)
	# plt.show()

	# Save JSON of preprocessed samples
	save_path = os.path.join(save_dir, this_sample_set_fname)

	samp_df.to_json(save_path, orient='records', indent=4)
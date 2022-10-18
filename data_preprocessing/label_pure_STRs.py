import json
import os

import numpy as np
import pandas as pd
import scipy.stats as stats
# import matplotlib.pyplot as plt
# import seaborn as sns
from tqdm import tqdm


def is_pure_repeat(seq, motif):
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


if __name__ == '__main__':
	# Options
	'''
	Args:
		data_dir: directory containing unlabeled samples JSON and statSTR 
			output TAB file, and where output labled_samples JSON will be
			saved.
		unlabeled_samp_fname: name of unlabeled samples JSON file
		statSTR_data_fname: name of statSTR output TAB file
	'''
	data_dir = os.path.join('..', 'data', 'heterozygosity')
	unlabeled_samp_fname = 'unlabeled_samples_dinucleotide_GRCh38_500_per_side.json'
	statSTR_data_fname = 'freqs_merged.tab'
	output_json_fname = 'labeled_samples_dinucleotide.json'
	remove_impure_repeats = True

	# Load target STRs to be labeled
	unlabeled_samp_path = os.path.join(data_dir, unlabeled_samp_fname)

	with open(unlabeled_samp_path) as fp:    
		samples = json.load(fp)

	# Load heterozygosity data from statSTR
	data_df = pd.read_csv(
		os.path.join(data_dir, statSTR_data_fname), 
		sep='\t', 
		header=0
	)
	data_df = data_df.dropna()
	print("total labeled data points:\t{}".format(len(data_df)))

	# # Plot distributions
	# sns.displot(data_df.het, kde=True)
	# hets = data_df.het.values.copy()
	# hets[hets == 0.0] = .001
	# sns.displot(np.log(hets), kde=True)
	# plt.show()

	# Get peak start and end locs by chromosome. Will be used to find 
	# overlap with STR regions.
	included_chroms = list(set(s['str_seq_name'].split(':')[0] for s in samples))

	chrom_dfs = dict()

	for chrom in included_chroms:
		chrom_dfs[chrom] = data_df[data_df.chrom == chrom]

	new_samples = []
	n_new_samples = 0
	n_called = []

	multi_match = 0

	# Label each STR sample with heterozygosity
	for samp in tqdm(samples):
		# Filter for pure repeats
		if remove_impure_repeats and not is_pure_repeat(samp['str_seq'], samp['motif']):
			continue

		# Match to statSTR label
		chrom, pos = samp['str_seq_name'].split(':')
		if 'complement' in pos:
			pos = pos.split(' ')[0]
			end, start = (int(val) for val in pos.split('-'))
		else:
			start, end = (int(val) for val in pos.split('-'))

		is_match = (chrom_dfs[chrom].start.values == start) \
					& (chrom_dfs[chrom].end.values - 1 == end)
	
		if is_match.sum() > 1:
			multi_match += 1
			print(f'multiple matches (skipped): {multi_match}')
			continue

		if is_match.any():
			range_ind = np.argwhere(is_match == True)[0]
			samp['heterozygosity'] = float(chrom_dfs[chrom].iloc[range_ind].het)
			assert not np.isnan(samp['heterozygosity'])
			samp['entropy'] = float(chrom_dfs[chrom].iloc[range_ind].entropy)
			samp['num_called'] = int(chrom_dfs[chrom].iloc[range_ind].numcalled)

			# acounts = chrom_dfs[chrom].iloc[range_ind].acount.values[0].split(',')
			# counts = np.array([int(c.split(':')[1]) for c in acounts])
			afreqs = chrom_dfs[chrom].iloc[range_ind].afreq.values[0].split(',')
			freqs = np.array([float(c.split(':')[1]) for c in afreqs])

			samp['minor_freq'] = float(1 - freqs.max())
			samp['minor_count'] = int(samp['minor_freq'] * samp['num_called'])

			new_samples.append(samp)
			n_called.append(samp['num_called'])

	print(len(new_samples))

	# Save labeled samples to main data dict
	samples_save_path = os.path.join(data_dir, output_json_fname)

	with open(samples_save_path, 'w') as fp:
		json.dump(new_samples, fp, indent=4)

	# sns.displot(np.array(n_called))
	# plt.show()
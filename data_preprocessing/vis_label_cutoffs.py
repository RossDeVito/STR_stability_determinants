import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':
	# Load labeled STRs to visualize
	samp_dir = os.path.join('..', 'data', 'heterozygosity')
	samp_fname = 'labeled_samples_dinucleotide.json'
	samp_path = os.path.join(samp_dir, samp_fname)
	
	show_plots = True
	save_plots = True
	save_dir = os.path.join('..', 'data', 'heterozygosity', 'label_cutoff_plots')

	with open(samp_path) as fp:    
		samples = json.load(fp)

	cutoff = 2000
	motif_types = ['GT', 'TG', 'CT', 'TC', 'AC', 'CA', 'AT', 'TA', 'AG', 'GA']
	max_copy_num = 6.5

	thresholds = [.001, .002, .0025, .005, .01, .02]

	df = pd.DataFrame(samples)
	df = df[df['num_called'] >= cutoff]
	df = df[df['num_copies'] <= max_copy_num]
	df = df[df['motif'].isin(motif_types)].reset_index(drop=True)

	df['non-zero minor_freq'] = df['minor_freq'] > 0

	# Make counts of samples by label with thresholds and plot
	split_counts = []
	for threshold in thresholds:
		mfrs = df['minor_freq'].values
		is_stable = (mfrs == 0)
		is_unstable = (mfrs[~is_stable] >= threshold)
		split_counts.append({
			'threshold': threshold,
			'class': 'stable',
			'count': np.sum(is_stable),
		})
		split_counts.append({
			'threshold': threshold,
			'class': 'unstable',
			'count': np.sum(is_unstable),
		})
		split_counts.append({
			'threshold': threshold,
			'class': 'unlabeled',
			'count': np.sum(~is_unstable),
		})
	split_counts = pd.DataFrame(split_counts)

	sns.barplot(x='threshold', y='count', hue='class', data=split_counts)
	plt.title('Label Counts by Threshold (max copy num: {:})'.format(max_copy_num))
	plt.xlabel('minor_freq threshold')
	if save_plots:
		plt.savefig(os.path.join(save_dir, 'cutoffs_mfr_maxcn{:}.png'.format(max_copy_num)))
	if show_plots:
		plt.show()

	# Plot MAF ECDF
	sns.displot(
		x='minor_freq', 
		hue='non-zero minor_freq', 
		data=df, 
		aspect=2.
	)
	plt.title('Minor Freq Distribtuion (max copy num: {:})'.format(max_copy_num))
	plt.xlim(-.01, .1)
	plt.tight_layout()
	for thresh in thresholds:
		plt.axvline(thresh, color='k', linestyle='--')

	if save_plots:
		plt.savefig(os.path.join(save_dir, 'mfr_dist_maxcn{:}.png'.format(max_copy_num)))
	if show_plots:
		plt.show()

	sns.displot(
		x='minor_freq', 
		hue='non-zero minor_freq',
		data=df, 
		kind='ecdf', 
		aspect=2.
	)
	plt.title('Minor Freq Distribtuion ECDF (max copy num: {:})'.format(max_copy_num))
	plt.tight_layout()
	plt.xlim(-.01, .2)
	for thresh in thresholds:
		plt.axvline(thresh, color='k', linestyle='--')
	if save_plots:
		plt.savefig(os.path.join(save_dir, 'mfr_dist_ecdf_maxcn{:}.png'.format(max_copy_num)))
	if show_plots:
		plt.show()
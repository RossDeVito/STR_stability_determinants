"""
Compute correlations between STR length and heterozygosity for all STRs
and plot length vs. heterozygosity for all.
"""

import os
from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm


def get_major_allel_freq(afreq_string):
	"""Get major allele frequency from allele frequency string."""
	afreqs = afreq_string.split(',')
	allele_lens = []
	allele_frac = []
	for afreq in afreqs:
		allele_len, allele_fract = afreq.split(':')
		allele_lens.append(len(allele_len))
		allele_frac.append(float(allele_fract))

	# Get major allele length
	return allele_lens[np.argmax(allele_frac)]


def custom_round(x, base=5):
    return int(base * round(float(x)/base))


if __name__ == '__main__':
	# Load heterozygosity data
	label_data_path = os.path.join(
		'..', '..', 'data', 'heterozygosity', 'freqs_merged.tab'
	)
	label_df = pd.read_csv(label_data_path, sep='\t')

	# Load motif info
	motif_info_dir = 'data'
	motif_info = []

	for chr in range(1, 23):
		motif_info_path = os.path.join(
			motif_info_dir, 'info_{}.txt'.format(chr)
		)
		chr_df = pd.read_csv(
			motif_info_path, 
			sep='\t', 
			header=None,
			names=['chrom', 'start', 'motif_len', 'motif']
		)
		motif_info.append(chr_df)

	motif_info = pd.concat(motif_info)

	# Merge
	merged_df = label_df.merge(motif_info, on=['chrom', 'start'])

	# Add new 'Motif Length' column with 1, 2, 3, 4, 5+
	merged_df['Motif Length'] = merged_df['motif_len'].apply(lambda x: 
		'1' if x == 1 else '2' if x == 2 else '3' if x == 3 else '4' if x == 4 else '5+'
	)

	# Add new 'Major Allele Length' column
	merged_df['Major Allele Length'] = merged_df['afreq'].apply(get_major_allel_freq)
	merged_df['copy_num'] = merged_df['Major Allele Length'] / merged_df['motif_len']
	merged_df = merged_df[merged_df['copy_num'] >= 2]

	# Compute correlations
	corrs = []
	for motif_len in tqdm(merged_df['Motif Length'].unique()):
		df = merged_df[merged_df['Motif Length'] == motif_len]
		pearson_corr, pearson_p = pearsonr(df['copy_num'], df['het'])
		spearman_corr, spearman_p = spearmanr(df['copy_num'], df['het'])
		corrs.append({
			'Motif Length': motif_len,
			'Pearson Correlation': pearson_corr,
			'Pearson p-value': pearson_p,
			'Spearman Correlation': spearman_corr,
			'Spearman p-value': spearman_p
		})

	corrs = pd.DataFrame(corrs).sort_values('Motif Length')
	print(corrs)

	# Round copy numbers to smooth
	round_to = 1	# will round to nearest round_to * n
	merged_df['Copy Number'] = merged_df['copy_num'].apply(
		partial(custom_round, base=2.0)
	)

	# Remove 'Motif Length' 'Copy Number' pairs with less than n samples
	min_samples = 4
	merged_df = merged_df.groupby(['Motif Length', 'Copy Number']).filter(
		lambda x: len(x) >= min_samples
	)

	# Plot
	percolors = ["gray","red","gold","blue","green","purple","brown"]
	sns.set_style('whitegrid')
	sns.set_context('paper', font_scale=1.5)
	sns.set_palette(percolors)

	sns.lineplot(
		data=merged_df,
		hue='Motif Length',
		hue_order=['1', '2', '3', '4', '5+'],
		x='Copy Number',
		y='het',
		errorbar=('ci', 99),
		n_boot=10,
		lw=2,
	)
	plt.ylabel('Heterozygosity')
	plt.xlabel('Major Allele Copy Number')
	plt.suptitle('STR Heterozygosity vs. Copy Number (99% CI)')
	plt.show()
import os
import json
# import multiprocessing as mp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm
from p_tqdm import p_map
from functools import partial


def score_fn_X_filter(samp_adjs, X_bases, agg_method='max'):
	"""
	Score function for samples that assigns a score of the max n_bases
	for cases that have an X value in X_bases (if X_bases is empty or None
	will not filter). If none meet the criteria returns 0.

	Args:
		sample_adjs: Dataframe containing 0 or more rows from pattern_df
		X_bases: List of bases to allow to be X for scoring. None or []
			means no filtering.
		agg_method: Method to aggregate the counts for flanking motifs.
			Options: 'max', 'sum'
	"""
	if X_bases is not None and X_bases != []:
		samp_adjs = samp_adjs[samp_adjs['X'].isin(X_bases)]
		
	if samp_adjs.shape[0] > 0:
		if agg_method == 'max':
			return samp_adjs.n_bases.max()
		elif agg_method == 'sum':
			return samp_adjs.n_bases.sum()
	else:
		return None


if __name__ == '__main__':
	__spec__ = None

	# Options
	label_version_dir = os.path.join(
		'find_patterns_output',
		# 'mfr0_005_mnc2000-m6_5',
		'mfr0_0025_mnc2000-m7_5',
		# 'mfr0_0_mnc2000-m7_5'
	)

	case = 1
	d = 0
	X_bases = ['A', 'C', 'G', 'T']
	agg_method = 'max'#'sum'
	num_cpus = 8

	do_plots = False
	inc_complements = False

	# Load data
	samp_df = pd.read_csv(os.path.join(label_version_dir, 'samples.csv'))
	pattern_df = pd.read_csv(os.path.join(label_version_dir, 'pattern_res.csv'))

	if not inc_complements:
		samp_df = samp_df[samp_df.complement == False]

	pattern_df = pattern_df.merge(
		samp_df[['sample_id', 'label', 'heterozygosity', 'entropy', 'minor_freq']],
		how='inner',
		on='sample_id'
	)

	pattern_df = pattern_df.sample(frac=.1)
	# pattern_df = pattern_df.head(100000)

	# reduce df
	df = pattern_df[(pattern_df['case'] == case) & (pattern_df['d'] == d)]

	# get scores
	s_ids = df.sample_id.unique()
	scores = p_map(
		partial(score_fn_X_filter, X_bases=X_bases, agg_method=agg_method),
		[df[df['sample_id'] == s_id] for s_id in df.sample_id.unique()],
		num_cpus=num_cpus
	)
	str_lens = [
		df[df['sample_id'] == s_id].str_len.mean() for s_id in df.sample_id.unique()
	]

	score_df = pd.DataFrame({
		'sample_id': s_ids, 
		'n_bases': scores,
		'str_len': str_lens
	})

	# Join labels
	score_df = score_df.merge(
		pattern_df[['sample_id', 'label', 'heterozygosity', 'entropy', 'minor_freq']],
		how='inner',
		on='sample_id'
	).dropna()

	# Get correlations
	corr_res = dict()
	corr_res['support'] = score_df.shape[0]

	# # corr_res[d]['mannwhitneyu'] = stats.mannwhitneyu(
	# # 	d_l1['n_bases'], 
	# # 	d_l0['n_bases']
	# # )
	# # corr_res[d]['mannwhitneyu_greater'] = stats.mannwhitneyu(
	# # 	d_l1['n_bases'], 
	# # 	d_l0['n_bases'], 
	# # 	alternative='greater'
	# # )
	# # corr_res[d]['mannwhitneyu_less'] = stats.mannwhitneyu(
	# # 	d_l1['n_bases'], 
	# # 	d_l0['n_bases'], 
	# # 	alternative='less'
	# # )
	corr_res['pointbiserialr'] = stats.pointbiserialr(
		score_df.n_bases,
		score_df.label
	)
	corr_res['pointbiserialr_strlen'] = stats.pointbiserialr(
		score_df.str_len,
		score_df.label
	)

	# try:
	# 	corr_res[d]['kruskal'] = stats.kruskal(
	# 		d_l1['n_bases'],
	# 		d_l0['n_bases']
	# 	)
	# except ValueError:
	# 	corr_res[d]['kruskal'] = 'invalid'

	corr_res['pearsonr_heterozygosity'] = stats.pearsonr(
		score_df.n_bases,
		score_df.heterozygosity
	)
	corr_res['pearsonr_strlen_heterozygosity'] = stats.pearsonr(
		score_df.str_len,
		score_df.heterozygosity
	)
	corr_res['spearmanr_heterozygosity'] = stats.spearmanr(
		score_df.n_bases,
		score_df.heterozygosity
	)
	corr_res['spearmanr_strlen_heterozygosity'] = stats.spearmanr(
		score_df.str_len,
		score_df.heterozygosity
	)

	corr_res['pearsonr_entropy'] = stats.pearsonr(
		score_df.n_bases,
		score_df.entropy
	)
	corr_res['pearsonr_strlen_entropy'] = stats.pearsonr(
		score_df.str_len,
		score_df.entropy
	)
	corr_res['spearmanr_entropy'] = stats.spearmanr(
		score_df.n_bases,
		score_df.entropy
	)
	corr_res['spearmanr_strlen_entropy'] = stats.spearmanr(
		score_df.str_len,
		score_df.entropy
	)

	corr_res['pearsonr_minor_freq'] = stats.pearsonr(
		score_df.n_bases,
		score_df.minor_freq
	)
	corr_res['pearsonr_strlen_minor_freq'] = stats.pearsonr(
		score_df.str_len,
		score_df.minor_freq
	)
	corr_res['spearmanr_minor_freq'] = stats.spearmanr(
		score_df.n_bases,
		score_df.minor_freq
	)
	corr_res['spearmanr_strlen_minor_freq'] = stats.spearmanr(
		score_df.str_len,
		score_df.minor_freq
	)

	# Save correlations
	complement_text = '_nc' if not inc_complements else ''
	save_path = os.path.join(
		label_version_dir, 
		f'case{case}_d{d}_Xbases{"".join(X_bases)}_agg{agg_method}{complement_text}.json'
	)
	with open(save_path, 'w') as f:
		json.dump(corr_res, f, indent=4)

	# Optional plots
	if do_plots:
		# sns.histplot(
		# 	x='n_bases',
		# 	hue='label',
		# 	data=score_df,
		# 	stat='count',
		# 	discrete=True,
		# 	common_norm=False,
		# 	multiple='layer',
		# 	# bw_adjust=2.0,
		# 	# common_norm=False,
		# )
		# # plt.savefig(os.path.join('plots', f'{plot_case}', f'hist_n_bases_count_d{d}.png'))
		# plt.show()

		sns.histplot(
			x='n_bases',
			hue='label',
			data=score_df,
			stat='probability',
			discrete=True,
			common_norm=False,
			multiple='layer',
		)
		# plt.savefig(os.path.join('plots', f'{plot_case}', f'hist_n_bases_prob_d{d}.png'))
		plt.show()
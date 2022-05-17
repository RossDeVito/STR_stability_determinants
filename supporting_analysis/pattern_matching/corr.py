import os
import json
import itertools
from re import S

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pingouin as pg
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


def get_biserial_correlations(score_df, str_len_corrs=True):
	""" 
	Get point biserial and rank biserial correlations for n_bases score
	and the STR length.
	"""
	# Point biserial
	pbs_nb = stats.pointbiserialr(
		score_df.n_bases,
		score_df.label
	)
	if str_len_corrs:
		pbs_strlen = stats.pointbiserialr(
			score_df.str_len,
			score_df.label
		)

	# Rank biserial
	n_bases_0 = score_df[score_df.label == 0].n_bases.values
	n_bases_1 = score_df[score_df.label == 1].n_bases.values
	rbs_nb = pg.mwu(n_bases_0, n_bases_1)

	if str_len_corrs:
		strlen_0 = score_df[score_df.label == 0].str_len.values
		strlen_1 = score_df[score_df.label == 1].str_len.values
		rbs_str_len = pg.mwu(strlen_0, strlen_1)

	ret_list = [{
		'correlation type': 'point-biserial',
		'feature': 'num. bases',
		'label': 'binary label',
		'correlation': pbs_nb.correlation, 
		'p-value': pbs_nb.pvalue
	}, {
		'correlation type': 'rank-biserial',
		'feature': 'num. bases',
		'label': 'binary label',
		'correlation': rbs_nb['RBC'][0],
		'p-value': rbs_nb['p-val'][0]	
	}]

	if str_len_corrs:
		ret_list.extend([{
			'correlation type': 'point-biserial',
			'feature': 'STR len.',
			'label': 'binary label',
			'correlation': pbs_strlen.correlation,
			'p-value': pbs_strlen.pvalue
		}, {
			'correlation type': 'rank-biserial',
			'feature': 'STR len.',
			'label': 'binary label',
			'correlation': rbs_str_len['RBC'][0],
			'p-value': rbs_str_len['p-val'][0]
		}])

	return ret_list


def get_continuous_correlations(score_df, label_col, str_len_corrs=False):
	pearson_nb = stats.pearsonr(
		score_df.n_bases,
		score_df[label_col]
	)
	spearman_nb = stats.spearmanr(
		score_df.n_bases,
		score_df[label_col]
	)
	ret_list = [{
		'correlation type': 'Pearson',
		'feature': 'num. bases',
		'label': label_col,
		'correlation': pearson_nb[0], 
		'p-value': pearson_nb[1]
	}, {
		'correlation type': 'Spearman',
		'feature': 'num. bases',
		'label': label_col,
		'correlation': spearman_nb[0],
		'p-value': spearman_nb[1]
	}]

	if str_len_corrs:
		pearson_strlen = stats.pearsonr(
			score_df.str_len,
			score_df[label_col]
		)
		spearman_strlen = stats.spearmanr(
			score_df.str_len,
			score_df[label_col]
		)
		ret_list.extend([{
			'correlation type': 'Pearson',
			'feature': 'STR len.',
			'label': label_col,
			'correlation': pearson_strlen[0],
			'p-value': pearson_strlen[1]
		}, {
			'correlation type': 'Spearman',
			'feature': 'STR len.',
			'label': label_col,
			'correlation': spearman_strlen[0],
			'p-value': spearman_strlen[1]
		}])

	return ret_list


def get_correlations(df, continuous_labels, agg_method, X_bases=None, 
		num_cpus=8, str_len_corrs=True):
	"""
	Get biserial and continuous correlations for STR lengt and the
	additional feature measured in n_bases.

	Args:
		df: Dataframe containing pattern_df format data of samples to use.
		continuous_labels: List of continuous label column names in df 
			to calculate correlations for.
		X_bases: List of bases to allow to be X for scoring. None or []
			no filtering and all used.
		num_cpus: Number of cpus to use for multiprocessing.
	"""
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
		df.drop_duplicates(subset=['sample_id'])[
			['sample_id', 'label', 'heterozygosity', 'entropy', 'minor_freq']],
		how='left',
		on='sample_id'
	).dropna()

	# Get correlations
	corr_res = dict()
	corr_res['support'] = score_df.shape[0]

	# Get biserial correlations
	all_corrs = get_biserial_correlations(score_df, str_len_corrs)

	# Get continuous correlations
	for label_col in continuous_labels:
		all_corrs.extend(get_continuous_correlations(score_df, label_col, str_len_corrs))

	return all_corrs


if __name__ == '__main__':
	__spec__ = None

	# Options
	label_version_dir = os.path.join(
		'find_patterns_output',
		'mfr0_005_mnc2000-m50',
		# 'mfr0_0025_mnc2000-m50',
		# 'mfr0_0_mnc2000-m64'
	)

	cases = [1, 2, 5]
	d_vals = [0, 1, 2]
	# max_str_lens = [13, 14, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 64]
	max_str_lens = [13, 15, 20, 25]#, 20, 35, 50, 64]
	X_bases = ['A', 'C', 'G', 'T']
	agg_methods = ['sum']
	continuous_labels = ['heterozygosity', 'entropy', 'minor_freq']

	do_plots = False
	inc_complements = False

	num_cpus = 10

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

	# pattern_df = pattern_df.sample(frac=.1)
	# pattern_df = pattern_df.head(100000)

	# Get correlations for different params
	all_corrs = []
	str_len_done = {l: False for l in max_str_lens}

	for case, d, max_str_len, agg_method in tqdm(
		itertools.product(cases, d_vals, max_str_lens, agg_methods),
		total=len(cases) * len(d_vals) * len(max_str_lens) * len(agg_methods)
	):
		if str_len_done[max_str_len]:
			do_str_len_corrs = False
		else:
			do_str_len_corrs = True
			str_len_done[max_str_len] = True

		# reduce df
		df = pattern_df[
			(pattern_df['case'] == case) 
			& (pattern_df['d'] == d)
			& (pattern_df['str_len'] <= max_str_len)
		]

		# Get correlations
		corr_res_df = pd.DataFrame(get_correlations(
			df, 
			continuous_labels, 
			X_bases=X_bases,
			agg_method=agg_method,
			num_cpus=num_cpus,
			str_len_corrs=do_str_len_corrs
		))
		
		corr_res_df['max_str_len'] = max_str_len
		corr_res_df['case'] = case
		corr_res_df['d'] = d
		corr_res_df['agg_method'] = agg_method
		corr_res_df['X_bases'] = 'ACGT' if (X_bases is None or X_bases == []) else ''.join(sorted(X_bases))

		all_corrs.append(corr_res_df)

	all_corrs_df = pd.concat(all_corrs, ignore_index=True)
	all_corrs_df.to_csv(
		os.path.join(label_version_dir, 'correlations_{}_cases{}_d{}_lens{}.csv'.format(
			'ACGT' if (X_bases is None or X_bases == []) else ''.join(sorted(X_bases)),
			'-'.join([str(v) for v in sorted(cases)]),
			'-'.join([str(v) for v in sorted(d_vals)]), 
			'-'.join([str(v) for v in sorted(max_str_lens)])
		)),
		index=False
	)
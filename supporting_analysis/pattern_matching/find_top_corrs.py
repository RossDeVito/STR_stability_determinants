import os

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


if __name__ == '__main__':
	# Load data
	data_path = os.path.join(
		'find_patterns_output', 
		'mfr0_005_mnc2000-m50',
		# 'correlations_ACGT_cases1-2-5_d0-1-2_lens12-13-14-15-20.csv'
		'correlations_by_motif_cases1-2-3-4-5-6_d0-1_lens13-15-20-25.csv'
	)
	df = pd.read_csv(data_path)
	df = df[df['feature'] == 'num. bases'].sort_values(
		'correlation', ascending=False
	).dropna().reset_index(drop=True)

	# Correct p-values
	df['p-value bonferroni'] = multipletests(pvals=df['p-value'], method='bonferroni')[1]
	df['p-value benjamini-hochberg'] = multipletests(pvals=df['p-value'], method='fdr_bh')[1]

	# To search for top params
	top_df = df.loc[df.reset_index().groupby(
		['correlation type', 'feature', 'case', 'd'])['correlation'].idxmax()
	].sort_values(
		'correlation', ascending=False
	).reset_index(drop=True)

	# Subset actual top params
	param_sets = [
	# {
	# 	'case': 5,
	# 	'd': 0,
	# 	'agg_method': 'sum',
	# 	'max_str_len': 15,
	# }, {
	# 	'case': 5,
	# 	'd': 0,
	# 	'agg_method': 'sum',
	# 	'max_str_len': 13,
	# }, {
	# 	'case': 1,
	# 	'd': 0,
	# 	'agg_method': 'sum',
	# 	'max_str_len': 13,
	# }, {
	# 	'case': 1,
	# 	'd': 0,
	# 	'agg_method': 'sum',
	# 	'max_str_len': 15,
	# }, {
	# 	'case': 2,
	# 	'd': 1,
	# 	'agg_method': 'sum',
	# 	'max_str_len': 13,
	# }, {
	# 	'case': 2,
	# 	'd': 1,
	# 	'agg_method': 'sum',
	# 	'max_str_len': 15,
	# }, 
	{
		'case': 3,
		'd': 1,
		'agg_method': 'sum',
		'max_str_len': 13,
	}, {
		'case': 3,
		'd': 1,
		'agg_method': 'sum',
		'max_str_len': 15,
	}, {
		'case': 3,
		'd': 0,
		'agg_method': 'sum',
		'max_str_len': 13,
	}, {
		'case': 3,
		'd': 0,
		'agg_method': 'sum',
		'max_str_len': 15,
	}, {
		'case': 4,
		'd': 1,
		'agg_method': 'sum',
		'max_str_len': 13,
	}, {
		'case': 4,
		'd': 1,
		'agg_method': 'sum',
		'max_str_len': 15,
	}, {
		'case': 6,
		'd': 1,
		'agg_method': 'sum',
		'max_str_len': 13,
	}, {
		'case': 6,
		'd': 1,
		'agg_method': 'sum',
		'max_str_len': 15,
	},
	]

	subsets = []

	for param_set in param_sets:
		subsets.append(df[
			(df['case'] == param_set['case']) &
			(df['d'] == param_set['d']) &
			(df['agg_method'] == param_set['agg_method']) &
			(df['max_str_len'] == param_set['max_str_len'])
		])
	
	final_df = pd.concat(subsets).reset_index(drop=True)

	# print by correlation type
	for corr_type in final_df['correlation type'].unique():
		print(corr_type)
		print(final_df[final_df['correlation type'] == corr_type].sort_values(
			'correlation', ascending=False))

	df[(df.max_str_len == 13) & (df.d == 0) & (df.case == 5)]

	df[(df.max_str_len == 15) & (df.feature == 'STR len.')]
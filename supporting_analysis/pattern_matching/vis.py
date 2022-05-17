import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


if __name__ == '__main__':
	samp_df = pd.read_csv('samples.csv')
	pattern_df = pd.read_csv('pattern_res.csv')

	samp_df = samp_df[samp_df['num_called'] > 100]

	pattern_df = pattern_df.merge(
		samp_df[['sample_id', 'label', 'binary_label']],
		how='inner',
		on='sample_id'
	)

	# sdf = pattern_df.sample(frac=.1)
	plot_case = 6
	d = 0
	c1_df = pattern_df[pattern_df['case'] == plot_case]
	c1_max_df = c1_df.groupby(['sample_id', 'case', 'd'], as_index=False).max()

	sns.histplot(
		x='n_bases',
		hue='binary_label',
		data=c1_max_df[c1_max_df['d'] == d],
		stat='count',
		discrete=True,
		common_norm=False,
		multiple='layer',
		# bw_adjust=2.0,
		# common_norm=False,
	)
	plt.savefig(os.path.join('plots', f'{plot_case}', f'hist_n_bases_count_d{d}.png'))
	plt.show()

	sns.histplot(
		x='n_bases',
		hue='binary_label',
		data=c1_max_df[c1_max_df['d'] == d],
		stat='probability',
		discrete=True,
		common_norm=False,
		multiple='layer',
	)
	plt.savefig(os.path.join('plots', f'{plot_case}', f'hist_n_bases_prob_d{d}.png'))
	plt.show()

	sns.displot(
		x='n_bases',
		hue='binary_label',
		data=c1_max_df[c1_max_df['d'] == d],
		kind='kde',
		bw_adjust=2.0,
		common_norm=False,
	)
	plt.savefig(os.path.join('plots', f'{plot_case}', f'kde_n_bases_count_d{d}.png'))
	plt.show()

	# sns.violinplot(
	# 	x='n_bases',
	# 	y='label',
	# 	hue='binary_label',
	# 	data=c1_max_df[c1_max_df['d'] == 0],
	# 	scale='count',
	# )
	# plt.show()

	# Tests
	df = c1_max_df[c1_max_df['d'] == 0]
	d_l0 = df[df['binary_label'] == 0]
	d_l1 = df[df['binary_label'] == 1]
	mw = stats.mannwhitneyu(d_l1['n_bases'], d_l0['n_bases'])
	mwg = stats.mannwhitneyu(d_l1['n_bases'], d_l0['n_bases'], alternative='greater')

"""Plot results of bootstrapping correlations."""

import os

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':
	data_path = os.path.join('saved_corrs', 'bootstrap_corrs_32.csv')
	target = 'heterozygosity'
	features = ['num_copies', 'all_dinuc_count']
	alpha = 1e-9

	# Load data
	data_df = pd.read_csv(data_path, header=0)

	# Plot just for STR length and all nearby dinuc repeats count
	data_df = data_df[data_df.target == target]
	data_df = data_df[data_df.feature.isin(features)]

	# Rename num_copies to Copy Number and all_dinuc_count to Dinucleotide 4-mer Count
	data_df.loc[data_df.feature == 'num_copies', 'feature'] = 'Copy Number'
	data_df.loc[data_df.feature == 'all_dinuc_count', 'feature'] = 'Dinuc. 4-mers'

	type_renaming_map = {
		'all': 'All',
		'AT': 'AT',
		'AC': 'AC/GT',
		'GT': 'AC/GT',
		'AG': 'AG/CT',
		'CT': 'AG/CT',
		'CG': 'CG/GC',
		'GC': 'CG/GC',
	}
	data_df['STR_type'] = data_df.STR_type.apply(lambda x: type_renaming_map[x])
	
	# Plot
	sns.set_style('whitegrid')
	sns.set_context('poster', font_scale=1)

	g = sns.barplot(
		data=data_df,
		x='STR_type',
		order=['All', 'AT', 'AC/GT', 'AG/CT'],
		hue='feature',
		# hue_order=['All', 'AT', 'AC/GT', 'AG/CT'],
		y='spearman_corr',
		estimator=np.median,
		errorbar=lambda x: (x.quantile(.005), x.quantile(.995))
	)
	g.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(.05))
	# g.set_xticklabels(g.get_xticklabels(), 
    #                       rotation=45, 
    #                       horizontalalignment='right')
	# g.set_xtickalignment('right')
	plt.legend(loc='upper right', title=None)
	# plt.suptitle(
	# 	f'12-15 bp Dinucleotide STR Correlations with {target.title()} (99% CI)',
	# 	y=.85,
	# 	x=.55
	# )
	plt.tight_layout()
	# plt.xlabel('STR Motif')
	# plt.ylabel('Spearman Coef.')
	plt.xlabel(None)
	plt.ylabel(None)
	# plt.tight_layout()
	plt.show()

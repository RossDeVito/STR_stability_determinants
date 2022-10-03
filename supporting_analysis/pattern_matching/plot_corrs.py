import os

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':
	# Options
	max_STR_lens = [13, 15, 25]
	to_plot = ['point-biserial', 'rank-biserial']#, 'heterozygosity']
	feats_to_plot = [	# Tuples formated (display_name, options_dict)
		('STR length', {'is_STR_len': True}),
		('Adj. dinucleotide\n repeat', {'case': 1, 'd': 0, 'agg_method': 'sum'}),
		('Adj. poly-base\n repeat', {'case': 2, 'd': 1, 'agg_method': 'sum'}),
		('Adj. every-other\n repeat', {'case': 5, 'd': 0, 'agg_method': 'sum'}),
	]
	p_val_threshold = 1e-8

	# Load data
	corr_dir = os.path.join('find_patterns_output', 'mfr0_005_mnc2000-m50')
	corr_fname = 'correlations_ACGT_cases1-2-5_d0-1-2_lens13-15-20-25.csv'
	corr_by_motif_fname = 'correlations_by_motif_cases1-2-5_d0-1_lens13-15-20-25.csv'

	df = pd.read_csv(os.path.join(corr_dir, corr_fname)).drop(columns='X_bases')
	df['motif'] = 'all'
	motif_df = pd.read_csv(os.path.join(corr_dir, corr_by_motif_fname))
	df = pd.concat([df, motif_df])
	df = df.dropna().reset_index(drop=True)

	# Remove duplicate tests on complements
	all_motif_types = ['AC/GT', 'AG/CT', 'AT', 'all']
	motif_map = {
		'all': 'all',
		'AC': 'AC/GT',
		'AG': 'AG/CT',
		'AT': 'AT',
		'CT': 'AG/CT',
		'GT': 'AC/GT',
	}
	df['motif'] = df['motif'].map(motif_map)
	df = df.drop_duplicates()

	# Correct p-values
	df['p-value bonferroni'] = multipletests(
		pvals=df['p-value'], 
		method='bonferroni'
	)[1]
	df['p-value benjamini-hochberg'] = multipletests(
		pvals=df['p-value'], 
		method='fdr_bh'
	)[1]

	# # Make plots
	# fig, axs = plt.subplots(
	# 	len(max_STR_lens), len(to_plot), sharey=True, squeeze=False
	# )
	# for ax in axs.flatten():
	# 	ax.grid(axis='y', zorder=0)

	# for row_idx, max_STR_len in enumerate(max_STR_lens):
	# 	# Reduce df
	# 	max_len_df = df[df['max_str_len'] == max_STR_len]

	# 	for col_idx,plot_type in enumerate(to_plot):
	# 		reduced_df = max_len_df[max_len_df['correlation type'] == plot_type]
	# 		rows_by_feat = []

	# 		for feat_name, feat_options in feats_to_plot:
	# 			if 'is_STR_len' in feat_options:
	# 				feat_df = reduced_df[reduced_df['feature'] == 'STR len.']
	# 			else:
	# 				feat_df = reduced_df[
	# 					(reduced_df['case'] == feat_options['case'])
	# 					& (reduced_df['d'] == feat_options['d'])
	# 					& (reduced_df['agg_method'] == feat_options['agg_method'])
	# 					& (reduced_df['feature'] == 'num. bases')
	# 				]
	# 			feat_df['Feature'] = feat_name
	# 			rows_by_feat.append(feat_df)
			
	# 		reduced_df = pd.concat(rows_by_feat)
	# 		reduced_df = reduced_df.rename(columns={
	# 			'correlation': 'Correlation',
	# 			'motif': 'Motif Type'
	# 		})
			
	# 		bar = sns.barplot(
	# 			x='Feature',
	# 			y='Correlation',
	# 			hue='Motif Type',
	# 			hue_order=all_motif_types,
	# 			data=reduced_df,
	# 			ax=axs[row_idx, col_idx],
	# 			zorder=3
	# 		)

	# 		# change shading for insignificant correlations
	# 		insignif_corr_vals = reduced_df[
	# 			reduced_df['p-value bonferroni'] > p_val_threshold
	# 		].Correlation.values

	# 		for j,this_bar in enumerate(bar.patches):
	# 			if this_bar.get_height() in insignif_corr_vals:
	# 				fill_color = bar.patches[j].get_facecolor()
	# 				bar.patches[j].set_edgecolor(fill_color)
	# 				bar.patches[j].set_facecolor("none")
	# 				bar.patches[j].set_hatch('/////')

	# 		# tweak labels and such
	# 		if col_idx == 0:
	# 			bar.set_ylabel('max STR len. = {}'.format(max_STR_len))
	# 		else:
	# 			bar.set_ylabel(None)

	# 		if row_idx == 0:
	# 			bar.set_title(plot_type.title())

	# 		if row_idx != len(max_STR_lens) - 1:
	# 			bar.set_xticklabels([])

	# 		bar.set_xlabel(None)
	# 		bar.axhline(y=0, color='gray', linestyle='-', lw=.5)			

	# for i,ax in enumerate(axs.flatten()):
	# 	ax.xaxis.set_tick_params(labelbottom=True)
	# 	ax.yaxis.set_tick_params(labelleft=True)
	# 	# ax.set_xticklabels(ax.get_xticklabels(), rotation=10, ha='right', rotation_mode='anchor')

	# 	# remove duplicate legends
	# 	if i == 0:
	# 		sns.move_legend(
	# 			ax,
	# 			'upper right',
	# 			# bbox_to_anchor=(.5, .99), 
	# 			ncol=len(all_motif_types) // 2,
	# 			# title=None,
	# 			frameon=True,
	# 		)
	# 	else:
	# 		ax.legend([],[], frameon=False)

		

	# fig.supylabel('Correlation')
	# fig.supxlabel('Features')
	# fig.suptitle('STR Stability Correlations')

	# plt.show()
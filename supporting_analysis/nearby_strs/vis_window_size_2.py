import os
from itertools import product

import pandas as pd
import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('macosx')
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':
	window_sizes = [32]
	max_str_lens = [13]
	base_data_file = 'sample_data_dinucleotide_withGC_mfr0_0_mnc2000'
	corr_type = 'spearman_corr'
	target = 'heterozygosity'	

	alpha = 1e-10

	# Load data
	all_data = []
	for window_size, max_len in product(window_sizes, max_str_lens):
		df = pd.read_csv(os.path.join(
			'saved_corrs', 
			f'ws_{window_size}_maxlen_{max_len}_{base_data_file}.csv'
		))
		df['window_size'] = window_size
		df['max_STR_len'] = max_len
		all_data.append(df)

	all_data = pd.concat(all_data, ignore_index=True)

	# Filter
	all_data = all_data[all_data['target'] == target]
	# all_data.loc[all_data.nearby_type == 'all_dinuc_count', 'nearby_type'] = 'Dinucleotide 4-mers'
	all_data['is_significant'] = all_data[corr_type[:-4] + 'pval_bonf'] < alpha
	all_data = all_data[all_data['is_significant']]

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
	nearby_type_renaming_map = {
		'all_dinuc_count': 'Dinuc. 4-mers',
		'STR_length': 'Copy Num.',
		'AT_count': 'AT_count',
		'AC_count': 'AC/GT_count',
		'GT_count': 'AC/GT_count',
		'AG_count': 'AG/CT_count',
		'CT_count': 'AG/CT_count',
		'CG_count': 'CG/GC_count',
		'GC_count': 'CG/GC_count',
		'AA_count': 'AA/TT_count',
		'TT_count': 'AA/TT_count',
		'CC_count': 'CC/GG_count',
		'GG_count': 'CC/GG_count',
	}
	all_data['STR_type'] = all_data.STR_type.apply(lambda x: type_renaming_map[x])
	all_data['nearby_type'] = all_data.nearby_type.apply(lambda x: nearby_type_renaming_map[x])
	all_data = all_data.drop_duplicates().reset_index(drop=True)

	# Plot
	sns.set_style('whitegrid')
	sns.set_context('poster', font_scale=1)
	# sns.set_palette('Set2')

	g = sns.barplot(
		data=all_data,
		x='nearby_type',
		# order=['All', 'AT', 'AC/GT', 'AG/CT', 'CG/GC'],
		hue='STR_type',
		hue_order=['All', 'AT', 'AC/GT', 'AG/CT'],
		# hue_order=[
		# 	'STR_length', 'Nearby dinucleotide repeats', 'AT_count', 'AC_count', 'GT_count', 
		# 	'AG_count', 'CT_count', 'AA_count', 'CC_count', 'GG_count', 'TT_count'
		# ],
		y=corr_type,
		width=.8,
		# legend_out=True,
		# legend=False,
	)
	g.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(.05))
	g.set_xticklabels(g.get_xticklabels(), 
                          rotation=25, 
                          horizontalalignment='right')
	# g.set_xtickalignment('right')
	plt.legend(loc='upper right', title='STR type')
	plt.suptitle(f'Significant Spearman Correlations with {target.title()} (Bonferroni-corrected p < {alpha})')
	plt.tight_layout()
	plt.xlabel(None)
	plt.ylabel('Spearman Coef.')

	# # rotate xticklabels and right align
	# for ax in g.axes.flat:
	# 	for label in ax.get_xticklabels():
	# 		label.set_rotation(35)
	# 		label.set_horizontalalignment('right')

	# plt.tight_layout()
	plt.gcf().set_dpi(300)
	plt.tight_layout()
	plt.show()

	# Minimal plot 
	all_data.loc[all_data.nearby_type == 'Dinucleotide 4-mers', 'nearby_type'] = 'Nearby Dinucleotide repeats'

	# Plot single bar plot
	sns.set_style('whitegrid')
	sns.set_context('talk', font_scale=1.)

	sns.barplot(
		data=all_data,
		x='STR_type',
		# order=['all', 'AT', 'AC', 'GT', 'AG', 'CT'],
		order=['All', 'AT', 'AC/GT', 'AG/CT'],
		hue='nearby_type',
		# hue_order=[
		# 	'STR_length', 'all_dinuc_count', 'AT_count', 'AC_count', 'GT_count',
		# 	'AG_count', 'CT_count', 'AA_count', 'CC_count', 'GG_count', 'TT_count'
		# ],
		hue_order=[
			'Copy Num.', 'Dinuc. 4-mers'
		],
		y=corr_type,
		# legend_out=True,
	)
	plt.gca().legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
	plt.suptitle("Significant Spearman Correlations with STR Entropy (Max STR len = {})".format(
		max_str_lens[0]
	))
	# plt.tight_layout(rect=[0.0, 0.01, .9, 1.0])
	# plt.tight_layout()
	plt.gcf().set_dpi(500)
	plt.show()

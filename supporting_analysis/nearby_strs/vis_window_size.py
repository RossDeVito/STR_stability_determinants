import os
from itertools import product

import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':
	window_sizes = [8, 16, 32, 64, 96, 128]
	max_str_len = 15
	base_data_file = 'sample_data_dinucleotide_mfr0_0_mnc2000'
	corr_types = ['pearson_corr', 'spearman_corr', 'point_biserial_corr', 'rank_biserial_corr']

	# Load data
	all_data = []
	for window_size in window_sizes:
		df = pd.read_csv(os.path.join(
			'saved_corrs', 
			f'ws_{window_size}_maxlen_{max_str_len}_{base_data_file}.csv'
		))
		df['window_size'] = window_size
		df['max_STR_len'] = max_str_len
		all_data.append(df)

	all_data = pd.concat(all_data, ignore_index=True)

	# # Plot
	# for corr_type in corr_types:
	# 	print(corr_type, flush=True)
	# 	sns.relplot(
	# 		data=all_data,
	# 		kind='line',
	# 		x='window_size',
	# 		y=corr_type,
	# 		col='target',
	# 		row='STR_type',
	# 		hue='nearby_type',
	# 		facet_kws={'margin_titles': True},
	# 		n_boot=0
	# 	)
	# 	print('plotted', flush=True)
	# 	plt.suptitle(f'{corr_type} vs. window size (max STR len. = {max_str_len}')
	# 	plt.tight_layout(rect=[0, 0.03, 1, 0.95])
	# 	plt.savefig(os.path.join('figures', 'by_window_size', f'ml{max_str_len}_{corr_type}_vs_window_size.png'))

	# plt.close()

import os

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns


if __name__ == '__main__':
	# Options
	patterns_output_dir = 'find_patterns_output'
	threshold_dirs = {
		'0.005': 'mfr0_005_mnc2000-m50',
		'0.0025': 'mfr0_0025_mnc2000-m50',
		'0.0': 'mfr0_0_mnc2000-m64'
	}

	label_version_dir = os.path.join(
		'find_patterns_output',
		# 'mfr0_005_mnc2000-m50',
		'mfr0_0025_mnc2000-m50',
		# 'mfr0_0_mnc2000-m64'
	)

	X_base_set = 'ACGT'
	agg_method = 'sum'

	# Load data
	all_dfs = []
	for thresh,thresh_dir in threshold_dirs.items():
		corr_df = pd.read_csv(os.path.join(
			patterns_output_dir,
			thresh_dir,
			'correlations_{}.csv'.format(X_base_set)
		))
		corr_df = corr_df[
			(corr_df['agg_method'] == agg_method)
			& (corr_df['label'] == 'binary label')
		]
		corr_df['Unstable Threshold'] = thresh
		all_dfs.append(corr_df)

	corr_df = pd.concat(all_dfs)

	# Make plot for each correlation type
	fig, axs = plt.subplots(1, len(corr_df['correlation type'].unique()), sharey=True, figsize=(10, 4))

	for i,corr_type in enumerate(corr_df['correlation type'].unique()):
		df = corr_df[corr_df['correlation type'] == corr_type].reset_index(drop=True)

		sns.lineplot(
			x='max_str_len',
			y='correlation',
			hue='Unstable Threshold',
			style='feature',
			data=df,
			ax=axs[i]
		)
		axs[i].set_title('{} Correlation'.format(corr_type.title()))
	plt.tight_layout()
	plt.savefig(os.path.join(patterns_output_dir, f'biserial_corrs_by_thresh.png'))
	plt.show()

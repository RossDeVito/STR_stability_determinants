import os

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns


if __name__ == '__main__':
	# Options
	label_version_dir = os.path.join(
		'find_patterns_output',
		# 'mfr0_005_mnc2000-m50',
		# 'mfr0_0025_mnc2000-m50',
		'mfr0_0_mnc2000-m64'
	)

	X_base_set = 'ACGT'
	agg_method = 'sum'
	# corr_with_labels = ['binary label', 'heterozygosity', 'entropy', 'minor_freq']
	corr_with_labels = ['heterozygosity', 'entropy', 'minor_freq']

	# Load data
	corr_df = pd.read_csv(os.path.join(
		label_version_dir, 
		'correlations_{}.csv'.format(X_base_set)
	))
	corr_df = corr_df[corr_df['agg_method'] == agg_method]
	corr_df = corr_df[corr_df['label'].isin(corr_with_labels)]

	# Make plot for corr with each value
	fig, axs = plt.subplots(1, len(corr_with_labels), sharey=True, figsize=(14, 4))

	for i,target_label in enumerate(corr_with_labels):
		df = corr_df[corr_df.label == target_label]
		# df['feature type'] = df.apply(lambda x: f'{x.feature} {x.agg_method}', axis=1)

		sns.lineplot(
			x='max_str_len',
			y='correlation',
			hue='correlation type',
			style='feature',
			data=df,
			ax=axs[i]
		)
		axs[i].set_title('Correlation with {}'.format(target_label))
	plt.tight_layout()
	plt.savefig(os.path.join(label_version_dir, f'corr_with_labels_{agg_method}.png'))
	plt.show()

	# Make plot for each correlation type
	fig, axs = plt.subplots(1, len(corr_df['correlation type'].unique()), sharey=True, figsize=(10, 4))

	for i,corr_type in enumerate(corr_df['correlation type'].unique()):
		df = corr_df[corr_df['correlation type'] == corr_type]

		sns.lineplot(
			x='max_str_len',
			y='correlation',
			hue='label',
			style='feature',
			data=df,
			ax=axs[i]
		)
		axs[i].set_title('{}'.format(corr_type))
	plt.tight_layout()
	plt.savefig(os.path.join(label_version_dir, f'corr_by_type_{agg_method}.png'))
	plt.show()

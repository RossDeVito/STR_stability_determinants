import os
import re
from itertools import combinations_with_replacement, product

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns

from data_modules import STRDataset


if __name__ == '__main__':
	# options
	str_len_min = 12
	str_len_max = 64

	target = 'entropy'

	type_renaming_map = {
		'all': 'All',
		'AT': 'AT',
		'AC': 'AC/GT',
		'GT': 'AC/GT',
		'AG': 'AG/CT',
		'CT': 'AG/CT',
	}

	training_params = {
		# Data File
		'data_dir': os.path.join('..', 'data', 'heterozygosity'),
		'data_fname': 'sample_data_dinucleotide_mfr0_0_mnc2000.json',

		# Data Module
		'min_copy_number': None,
		'max_copy_number': 64,
		'incl_STR_feat': True,
		'min_boundary_STR_pos': 6,
		'max_boundary_STR_pos': 6,
		'window_size': str_len_max,
		'bp_dist_units': 1000.0,
	}

	# Load with DataModule
	data_path = os.path.join(
		'..', '..', 'data', 'heterozygosity', training_params['data_fname']
	)
	data_module = STRDataset(
		data_path,
		split_name=None,
		incl_STR_feat=training_params['incl_STR_feat'],
		min_boundary_STR_pos=training_params['min_boundary_STR_pos'],
		max_boundary_STR_pos=training_params['max_boundary_STR_pos'],
		window_size=training_params['window_size'],
		min_copy_num=training_params['min_copy_number'],
		max_copy_num=training_params['max_copy_number'],
		bp_dist_units=training_params['bp_dist_units'],
		return_strings=True,
		return_data=True
	)
	samples = data_module.data
	samples['alpha_motif'] = samples.motif.apply(lambda x: ''.join(sorted(x)))

	# Plot lenght vs. heterozygosity
	samples['STR Length'] = samples.num_copies * 2
	samples['STR Motif'] = 'All'

	samples_copy = samples.copy()
	samples_copy['STR Motif'] = samples_copy.alpha_motif.apply(
		lambda x: type_renaming_map[x]
	)

	doubled_samples = pd.concat([samples, samples_copy], ignore_index=True)
	doubled_samples = doubled_samples.rename(columns={'heterozygosity': 'Heterozygosity'})

	sns.lineplot(
		data=doubled_samples,
		x='STR Length',
		y='Heterozygosity',
		hue='STR Motif',
		ci=99
	)
	plt.suptitle('Heterozygosity vs. STR Length (99% CI)')
	plt.show()

	# Get Pearson and Spearman correlations at each max STR length
	core_STR_motifs = ['all', *samples.alpha_motif.unique()]
	corrs = []

	for str_len in tqdm(range(str_len_min, str_len_max + 1)):

		max_len_samples = samples[samples.num_copies * 2 <= str_len]
		for core_type in core_STR_motifs:
			if core_type == 'all':
				core_samples = max_len_samples
			else:
				core_samples = max_len_samples[max_len_samples.alpha_motif == core_type]

			# Get correlations
			label_vals = core_samples[target].values
			str_lengths = core_samples.num_copies.values * 2

			pearson_corr = stats.pearsonr(
				str_lengths,
				label_vals
			)
			spearman_corr = stats.spearmanr(
				str_lengths,
				label_vals
			)

			corrs.append({
				'STR_type': core_type,
				'corr_type': 'Pearson',
				'corr': pearson_corr[0],
				'pval': pearson_corr[1],
				'max_STR_len': str_len,
			})
			corrs.append({
				'STR_type': core_type,
				'corr_type': 'Spearman',
				'corr': spearman_corr[0],
				'pval': spearman_corr[1],
				'max_STR_len': str_len,
			})

	# Agg
	corrs = pd.DataFrame(corrs)
	corrs = corrs[corrs.corr_type == 'Spearman']

	corrs['STR_type'] = corrs.STR_type.apply(lambda x: type_renaming_map[x])
	corrs = corrs.drop_duplicates().reset_index(drop=True)

	# Plot
	sns.set_theme(style='whitegrid')

	sns.lineplot(
		data=corrs,
		x='max_STR_len',
		y='corr',
		hue='STR_type'
	)
	plt.suptitle('Spearman Correlation with STR Length')
	plt.xlabel('Max STR Length')
	plt.ylabel('Spearman Correlation')
	plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
	plt.show()
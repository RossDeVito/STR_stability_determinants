"""Find correlations with length and nearby STRs with bootstapping and save."""

import os
import re
from itertools import combinations_with_replacement, product

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
# import pingouin as pg

from data_modules import STRDataset


if __name__ == '__main__':
	# options
	window_size = 32
	max_str_len = 15

	# bootstrap options
	n_bootstraps = 1000
	bootstrap_frac = 0.7

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
		'window_size': window_size,
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

	# Set max length
	samples = samples[samples.num_copies * 2 <= max_str_len]
	samples = samples.rename(columns={'minor_count': 'n_minor_allel'})
	samples['alpha_motif'] = samples.motif.apply(lambda x: ''.join(sorted(x)))

	# Reduce pre and post seqs to window_size number of adjacent bases
	samples['pre_seq'] = samples['pre_seq'].apply(lambda x: x[-window_size:])
	samples['post_seq'] = samples['post_seq'].apply(lambda x: x[:window_size])

	# For each dinucletide STR type, could number of occurances in each sample's
	# pre and post seqs
	for str_bases in tqdm(
			combinations_with_replacement(['A', 'C', 'G', 'T'], 2),
			desc='Counting nearby STRs'
		):
		sorted_motif = ''.join(sorted(str_bases))

		# Get number of occurances
		motif_counts = samples.pre_seq.apply(
			lambda x: len(re.findall(f'(?={sorted_motif}{sorted_motif})', x))
		)
		motif_counts += samples.post_seq.apply(
			lambda x: len(re.findall(f'(?={sorted_motif}{sorted_motif})', x))
		)

		# Get number for other ordering of bases if not homopolymer
		reversed_motif = ''.join(sorted_motif[::-1])
		if reversed_motif != sorted_motif:
			motif_counts += samples.pre_seq.apply(
				lambda x: len(re.findall(f'(?={reversed_motif}{reversed_motif})', x))
			)
			motif_counts += samples.post_seq.apply(
				lambda x: len(re.findall(f'(?={reversed_motif}{reversed_motif})', x))
			)

		# Add to samples
		samples[f'{sorted_motif}_count'] = motif_counts

	# Add together counts for all dinucleotide STRs
	samples['all_dinuc_count'] = samples[
		[k for k in samples.keys() if '_count' in k]
	].sum(axis=1)

	# Get correlations between stability and nearby STR counts for all
	# types of nearby and STR dinucleotide patters, and overall
	corrs = []
	core_STR_motifs = ['all', *samples.alpha_motif.unique()]
	features = ['num_copies', 'all_dinuc_count']
	target_labels = ['label', 'heterozygosity', 'entropy', 'minor_freq']

	# Randomly subsample and get correlations
	for _ in tqdm(range(n_bootstraps), desc='Bootstrapping', total=n_bootstraps):
		# Randomly subsample
		samples_sub = samples.sample(frac=bootstrap_frac, replace=False)

		for core_STR_motif, feature, target_label in product(
				core_STR_motifs, features, target_labels
			):
			# Get data for this motif and feature
			if core_STR_motif == 'all':
				data = samples_sub
			else:
				data = samples_sub[samples_sub.alpha_motif == core_STR_motif]
			data = data[[feature, target_label]]

			# Get correlations
			pearson_corr = stats.pearsonr(
				data[feature].values, data[target_label].values
			)
			spearman_corr = stats.spearmanr(
				data[feature].values, data[target_label].values
			)
			
			corrs.append({
				'STR_type': core_STR_motif,
				'feature': feature,
				'target': target_label,
				'pearson_corr': pearson_corr[0],
				'pearson_pval': pearson_corr[1],
				'spearman_corr': spearman_corr[0],
				'spearman_pval': spearman_corr[1]
			})

	# Save
	corrs = pd.DataFrame(corrs)
	corrs.to_csv(
		os.path.join('saved_corrs', f'bootstrap_corrs_{window_size}.csv'),
		index=False
	)
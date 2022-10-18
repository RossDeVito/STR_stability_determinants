import os
import re
from itertools import combinations_with_replacement, product

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
import pingouin as pg

from data_modules import STRDataset


if __name__ == '__main__':
	# options
	window_size = 16
	max_str_len = 20

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
	nearby_motifs = [k for k in samples.keys() if '_count' in k]
	target_labels = ['label', 'heterozygosity', 'entropy', 'minor_freq']

	for core_type, nearby_type, target in tqdm(
		product(core_STR_motifs, nearby_motifs, target_labels),
		total=len(core_STR_motifs) * len(nearby_motifs) * len(target_labels),
		desc='Calculating correlations'
	):
		if core_type == 'all':
			core_STR_samples = samples
		else:
			core_STR_samples = samples[samples.alpha_motif == core_type]
		label_vals = core_STR_samples[target].values
		nearby_motif_counts = core_STR_samples[nearby_type].values

		pearson_corr = stats.pearsonr(
			nearby_motif_counts,
			label_vals
		)
		spearman_corr = stats.spearmanr(
			nearby_motif_counts,
			label_vals
		)

		if target == 'label':
			point_biserial_corr = stats.pointbiserialr(
				nearby_motif_counts,
				label_vals
			)

			# Rank biserial
			n_motifs_0 = nearby_motif_counts[label_vals == 0]
			n_motifs_1 = nearby_motif_counts[label_vals == 1]
			rank_biserial_corr = pg.mwu(n_motifs_0, n_motifs_1)
		else:
			point_biserial_corr = None
			rank_biserial_corr = None
		
		corrs.append({
			'STR_type': core_type,
			'nearby_type': nearby_type,
			'target': target,
			'pearson_corr': pearson_corr[0],
			'pearson_pval': pearson_corr[1],
			'spearman_corr': spearman_corr[0],
			'spearman_pval': spearman_corr[1],
			'point_biserial_corr': point_biserial_corr.correlation if point_biserial_corr is not None else None,
			'point_biserial_pval': point_biserial_corr.pvalue if point_biserial_corr is not None else None,
			'rank_biserial_corr': rank_biserial_corr['RBC'][0] if rank_biserial_corr is not None else None,
			'rank_biserial_pval': rank_biserial_corr['p-val'][0] if rank_biserial_corr is not None else None,
		})

	# Get correlation with STR length
	for core_type, target in tqdm(
		product(core_STR_motifs, target_labels),
		desc='Calculating correlations with STR length',
	):
		if core_type == 'all':
			core_STR_samples = samples
		else:
			core_STR_samples = samples[samples.alpha_motif == core_type]
		label_vals = core_STR_samples[target].values

		pearson_corr = stats.pearsonr(
			core_STR_samples.num_copies.values,
			label_vals
		)
		spearman_corr = stats.spearmanr(
			core_STR_samples.num_copies.values,
			label_vals
		)

		if target == 'label':
			point_biserial_corr = stats.pointbiserialr(
				core_STR_samples.num_copies.values,
				label_vals
			)

			# Rank biserial
			n_motifs_0 = core_STR_samples.num_copies.values[label_vals == 0]
			n_motifs_1 = core_STR_samples.num_copies.values[label_vals == 1]
			rank_biserial_corr = pg.mwu(n_motifs_0, n_motifs_1)
		else:
			point_biserial_corr = None
			rank_biserial_corr = None

		corrs.append({
			'STR_type': core_type,
			'nearby_type': 'STR_length',
			'target': target,
			'pearson_corr': pearson_corr[0],
			'pearson_pval': pearson_corr[1],
			'spearman_corr': spearman_corr[0],
			'spearman_pval': spearman_corr[1],
			'point_biserial_corr': point_biserial_corr.correlation if point_biserial_corr is not None else None,
			'point_biserial_pval': point_biserial_corr.pvalue if point_biserial_corr is not None else None,
			'rank_biserial_corr': rank_biserial_corr['RBC'][0] if rank_biserial_corr is not None else None,
			'rank_biserial_pval': rank_biserial_corr['p-val'][0] if rank_biserial_corr is not None else None,
		})

	# Correct p-values
	corrs = pd.DataFrame(corrs)
	corrs['pearson_pval_bonf'] = corrs['pearson_pval'] * len(corrs)
	corrs['spearman_pval_bonf'] = corrs['spearman_pval'] * len(corrs)
	corrs['point_biserial_pval_bonf'] = corrs['point_biserial_pval'] * len(corrs.dropna())
	corrs['rank_biserial_pval_bonf'] = corrs['rank_biserial_pval'] * len(corrs.dropna())

	# Save correlations
	corrs = corrs.sort_values('spearman_corr', ascending=False)
	corrs.to_csv(
		os.path.join(
			'saved_corrs',
			'ws_{:}_maxlen_{:}_{:}.csv'.format(
				window_size,
				max_str_len,
				training_params['data_fname'].split(".")[0]
			)
		), 
		index=False
	)

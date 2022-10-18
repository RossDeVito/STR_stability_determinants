import os

import numpy as np
import pandas as pd
from tqdm import tqdm, trange

from data_modules import STRDataset


if __name__ == '__main__':
	max_copy_num_map = {
		'mfr0_005': 6.5,
		'mfr0_0025': 7.5,
	}

	# options
	label_version = 'mfr0_005'
	save_dir = 'fasta_files'
	window_size = 64
	inc_STR_cn = 0 #  number of STR copies to include in FASTA
	inc_STR_dummies = False # if included STR bp should be represented as 'XY' instead of real bases

	training_params = {
		# Data File
		'data_dir': os.path.join('..', 'data', 'heterozygosity'),
		'data_fname': f'sample_data_dinucleotide_{label_version}_mnc2000.json',

		# Data Module
		'min_copy_number': None,
		'max_copy_number': max_copy_num_map[label_version],
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

	# Sort samples by STR start/end motifs
	samples['alpha_motif'] = samples.motif.apply(lambda x: ''.join(sorted(x)))
	print(samples.alpha_motif.value_counts())

	# Save FASTA files
	fasta_verison_dir = os.path.join(
		save_dir,
		f'{label_version}_w{window_size}_cn{inc_STR_cn}_both_flanks'
	)
	if not os.path.exists(fasta_verison_dir):
		os.makedirs(fasta_verison_dir)

	for motif, motif_samples in tqdm(samples.groupby('alpha_motif')):
		# Make fasta file for negatives
		neg_samples = motif_samples[motif_samples.label == 0]
		fasta_file = os.path.join(
			fasta_verison_dir,
			'{}_{}_fasta.fa'.format(motif, 0)
		)
		with open(fasta_file, 'w') as f:
			for i, row in neg_samples.iterrows():
				pre_seq = row['pre_seq'][-window_size:]
				post_seq = row['post_seq'][:window_size]
				f.write('>{}\n{}\n'.format(row.HipSTR_name + 'pre', pre_seq))
				f.write('>{}\n{}\n'.format(row.HipSTR_name + 'post', post_seq))

		# Make fasta file for positives
		pos_samples = motif_samples[motif_samples.label == 1]
		fasta_file = os.path.join(
			fasta_verison_dir,
			'{}_{}_fasta.fa'.format(motif, 1)
		)
		with open(fasta_file, 'w') as f:
			for i, row in pos_samples.iterrows():
				pre_seq = row['pre_seq'][-window_size:]
				post_seq = row['post_seq'][:window_size]
				f.write('>{}\n{}\n'.format(row.HipSTR_name + 'pre', pre_seq))
				f.write('>{}\n{}\n'.format(row.HipSTR_name + 'post', post_seq))
				

import os

import numpy as np
import pandas as pd
from tqdm import tqdm, trange

from data_modules import STRDataset


def sort_by_motif_start_end(samples, motif_types, motif_len=2):
	""" Sort samples into pre and post STR seqs that boarder different
	start and end bases of STR seqs, and in this version by binary 
	label too. 
	"""
	sorted_samples = {
		motif: {'pre': {0:[],1:[]}, 'post': {0:[],1:[]}, 'both': {0:[],1:[]}} for motif in motif_types
	}
	
	for i,s in tqdm(samples.iterrows(), total=len(samples)):
		str_start = s['str_seq'][:motif_len]
		str_end = s['str_seq'][-motif_len:]
		sorted_samples[str_start]['pre'][s.label].append(s)
		sorted_samples[str_end]['post'][s.label].append(s)

	return sorted_samples


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
	# motif_types = ['CA', 'AC']
	motif_types = samples.motif.unique().tolist()
	sorted_samples = sort_by_motif_start_end(
		samples, motif_types, motif_len=len(motif_types[0])
	)
	counts_by_str_type = {
		k: {0:(len(v['pre'][0]), len(v['post'][0])), 1:(len(v['pre'][1]), len(v['post'][1]))} 
			for k,v in sorted_samples.items()
	}
	print(counts_by_str_type)

	# Write fasta files for each label for each STR start/end motif
	if inc_STR_cn == 0:
		fasta_verison_dir = os.path.join(
			save_dir,
			f'{label_version}_w{window_size}_cn{inc_STR_cn}'
		)
	elif inc_STR_dummies:
		fasta_verison_dir = os.path.join(
			save_dir,
			f'{label_version}_w{window_size}_cn{inc_STR_cn}_dummies'
		)
	else:
		fasta_verison_dir = os.path.join(
			save_dir,
			f'{label_version}_w{window_size}_cn{inc_STR_cn}'
		)

	if not os.path.exists(fasta_verison_dir):
		os.makedirs(fasta_verison_dir)

	for motif, by_pos_dict in sorted_samples.items():
		for pos, by_label_dict in by_pos_dict.items():
			for label, samps in by_label_dict.items():
				# Make fasta file
				fasta_file = os.path.join(
					fasta_verison_dir,
					'{}_{}_{}_fasta.fa'.format(
						motif, pos, label,
					)
				)
				with open(fasta_file, 'w') as f:
					if inc_STR_cn == 0:
						for s in samps:
							if pos == 'pre':
								seq = s['pre_seq'][-window_size:]
							else:
								seq = s['post_seq'][:window_size]
							f.write('>{}\n{}\n'.format(s.HipSTR_name, seq))
					elif inc_STR_dummies:
						for s in samps:
							if pos == 'pre':
								seq = s['pre_seq'][-window_size:] + 'XY'*inc_STR_cn
							else:
								seq = 'YX'*inc_STR_cn + s['post_seq'][:window_size]
							f.write('>{}\n{}\n'.format(s.HipSTR_name, seq)) 
					else:
						for s in samps:
							if pos == 'pre':
								seq = s['pre_seq'][-window_size:] + s.str_seq[:2*inc_STR_cn]
							else:
								seq = s.str_seq[-2*inc_STR_cn:][::-1] + s['post_seq'][:window_size]
							f.write('>{}\n{}\n'.format(s.HipSTR_name, seq))
"""Finds target STR types in BED output from HipSTR and extracts
surrounding sequence from reference genome. Saves resuting samples
to a JSON file in main data dir.
"""
import os
import json

import numpy as np
import pandas as pd
# from statsmodels.distributions.empirical_distribution import ECDF
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from pyfaidx import Fasta


if __name__ == '__main__':
	# Options
	'''
	Args:
		target_motifs: list of motifs to extract, or one of {'dinucleotide'}
		str_region_bed_fname: name of HipSTR reference STR region BED file
		ref_fasta_path: path to reference genome fasta file
		n_per_side: number of additional allels on either side of STR 
			seq to save
		output_json_fname: name of output unlabeled samples JSON file, or None
			to automatically name based on target_motifs and n_per_side
		output_save_dir: directory to save unlabled samples to
	'''
	target_STR_motifs = 'dinucleotide'
	str_region_bed_fname = 'GRCh38.hipstr_reference.bed.gz'
	ref_fasta_path = '../data/human-references/GRCh38_full_analysis_set_plus_decoy_hla.fa'
	n_per_side = 500
	output_json_fname = None
	samples_save_dir = os.path.join('..', 'data', 'heterozygosity_v12_22_filtered')

	# Load STR region BED
	str_region_bed_path = os.path.join('..', 'data', 'HipSTR-references', 
										str_region_bed_fname)

	print("Loading bed data...")
	str_regions = pd.read_csv(
		str_region_bed_path, 
		sep='\t', 
		names=['chr', 'start', 'stop', 'motif_len', 
				'num_copies', 'str_name', 'motif'],
		low_memory=False, # because chr field is mixed type
	)
	print("total STRs:\t{}".format(len(str_regions)))

	# Filter down to relevant motifs
	if isinstance(target_STR_motifs, str):
		if target_STR_motifs == 'dinucleotide':
			target_motifs = [
				'GT', 'TG', 'CT', 'TC', 'AC', 'CA', 
				'AT', 'TA', 'AG', 'GA', 'GC', 'CG'
			]
		else:
			raise ValueError('target_motifs must be a list of motifs or "dinucleotide"')
	else:
		target_motifs = target_STR_motifs

	str_regions = str_regions[str_regions.motif.isin(target_motifs)]

	print("found target regions:")
	print(str_regions.motif.value_counts())

	# Load reference genome
	ref_genome = Fasta(ref_fasta_path)

	# Extract sequence around each relevant STR region
	samples = []

	for _, region in tqdm(str_regions.iterrows(), 
							desc='Getting sample regions',
							total=len(str_regions)):
		chr_str = 'chr{}'.format(region.chr)
		str_seq = ref_genome[chr_str][region.start-1 : region.stop]
		pre_seq = ref_genome[chr_str][region.start-1-n_per_side : region.start-1]
		post_seq = ref_genome[chr_str][region.stop : region.stop+n_per_side]
		full_seq = ref_genome[chr_str][region.start-1-n_per_side : region.stop+n_per_side]
		
		assert pre_seq.seq + str_seq.seq + post_seq.seq == full_seq.seq
		assert pre_seq.complement.seq + str_seq.complement.seq + post_seq.complement.seq == full_seq.complement.seq

		# add sequence
		samples.append({
			'HipSTR_name': region.str_name,
			'complement': False,
			'motif': region.motif,
			'motif_len': region.motif_len,
			'num_copies': region.num_copies,
			'str_seq': str_seq.seq,
			'str_seq_name': str_seq.fancy_name,
			'pre_seq': pre_seq.seq,
			'pre_seq_name': pre_seq.fancy_name,
			'post_seq': post_seq.seq,
			'post_seq_name': post_seq.fancy_name,
			'full_seq': full_seq.seq,
			'full_seq_name': full_seq.fancy_name,
			'n_per_side': n_per_side
		})

		# add sequence complement
		samples.append({
			'HipSTR_name': region.str_name,
			'complement': True,
			'motif': str_seq.reverse.complement.seq[:region.motif_len],
			'motif_len': region.motif_len,
			'num_copies': region.num_copies,
			'str_seq': str_seq.reverse.complement.seq,
			'str_seq_name': str_seq.reverse.complement.fancy_name,
			'pre_seq': post_seq.reverse.complement.seq,
			'pre_seq_name': post_seq.reverse.complement.fancy_name,
			'post_seq': pre_seq.reverse.complement.seq,
			'post_seq_name': pre_seq.reverse.complement.fancy_name,
			'full_seq': full_seq.reverse.complement.seq,
			'full_seq_name': full_seq.reverse.complement.fancy_name,
			'n_per_side': n_per_side
		})

	# Save unlabeled samples to main data dict
	if output_json_fname is None:
		if isinstance(target_STR_motifs, str):
			output_json_fname = f'unlabeled_samples_{target_STR_motifs}_GRCh38_{n_per_side}_per_side.json'
		else:
			motifs = '-'.join(sorted(target_STR_motifs))
			output_json_fname = f'unlabeled_samples_{motifs}_GRCh38_{n_per_side}_per_side.json'

	samples_save_path = os.path.join(samples_save_dir, output_json_fname)

	with open(samples_save_path, 'w') as fp:
		json.dump(samples, fp, indent=4)
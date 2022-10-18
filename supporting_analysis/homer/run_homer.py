import gc
import os
import re
import subprocess

from tqdm import tqdm


def make_HOMER_cmd(fasta_fname, background_fasta_fname, output_dir, 
					motif_lens, gc_norm, num_cpus):
	if gc_norm:
		return "findMotifs.pl {} fasta {} -fasta {} -p {} -len {}".format(
			fasta_fname, output_dir, background_fasta_fname, num_cpus, 
			','.join([str(l) for l in motif_lens])
		)
	else:
		return "findMotifs.pl {} fasta {} -fasta {} -p {} -len {} -noweight".format(
			fasta_fname, output_dir, background_fasta_fname, num_cpus, 
			','.join([str(l) for l in motif_lens])
		)

if __name__ == '__main__':
	# Options
	label_version_dir = 'mfr0_005_w64_cn0_both_flanks'
	fasta_dir = os.path.join('fasta_files', label_version_dir)
	num_cpus = 10
	motif_lens = [4,5,6,7,8,9,10,11,12]
	gc_norm = False

	show_in_shell = False

	# Create output directory for HOMER w/ current params
	output_dir = os.path.join(
		'homer_output', 
		label_version_dir, 
		'_'.join([str(x) for x in motif_lens]) + '_gc_norm' if gc_norm else ''
	)
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	# See what files are in fasta_dir
	fasta_files = os.listdir(fasta_dir)

	for fname in tqdm(fasta_files, total=len(fasta_files)):
		# Make dir for output for motif/pos/label combination
		case_output_dir = os.path.join(output_dir, fname.split('_fasta')[0])
		if not os.path.exists(case_output_dir):
			os.makedirs(case_output_dir)

		# get background file name
		if '0' in fname:
			background_fname = fname.replace('0', '1')
		else:
			background_fname = fname.replace('1', '0')
		assert fname != background_fname

		# Run HOMER
		subprocess.run(
			make_HOMER_cmd(
				os.path.join(fasta_dir, fname), 
				os.path.join(fasta_dir, background_fname), 
				case_output_dir, motif_lens, gc_norm, num_cpus
			),
			shell=True,
			check=True,
			stdout=subprocess.DEVNULL,
			stderr=subprocess.DEVNULL,
		)

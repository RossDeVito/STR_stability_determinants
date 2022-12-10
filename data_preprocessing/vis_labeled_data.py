import os
import json

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':
	# Load labeled STRs to visualize
	samp_dir = os.path.join('..', 'data', 'heterozygosity_v12_22_filtered')
	samp_fname = 'labeled_samples_dinucleotide.json'
	samp_path = os.path.join(samp_dir, samp_fname)
	
	show_plots = True
	save_plots = True
	save_dir = os.path.join(samp_dir, 'plots')

	with open(samp_path) as fp:    
		samples = json.load(fp)

	# Plot dist of number of samples called
	num_samples = np.array([s['num_called'] for s in samples])
	sns.displot(num_samples, kind='ecdf')
	plt.title('Number of samples called ECDF')
	plt.tight_layout()
	if save_plots:
		plt.savefig(os.path.join(save_dir, 'num_called_dist.png'))
	if show_plots:
		plt.show()

	cutoffs = [1000, 1500, 2000, 2500]
	for cutoff in cutoffs:
		print(f'cutoff: {cutoff}\texcludes: {(num_samples < cutoff).sum()}/{len(num_samples)}\tuses: {(num_samples >= cutoff).sum()} {(num_samples >= cutoff).sum()/len(num_samples)*100:.2f}%')

	# Plot the distibution of the heterozygosity values
	hets = np.array([s['heterozygosity'] for s in samples])
	het_data = {'heterozygosity': hets, 'non-zero heterozygosity': (hets == 0)}
	bins = np.linspace(0,1,101)
	sns.displot(x='heterozygosity', hue='non-zero heterozygosity', data=het_data, bins=bins)
	plt.title('Heterozygosity Distribtuion (bin width: {:f})'.format(bins[1]-bins[0]))
	plt.tight_layout()
	if save_plots:
		plt.savefig(os.path.join(save_dir, 'het_dist_bw{:f}.png'.format(bins[1]-bins[0])))
	if show_plots:
		plt.show()

	bins = np.linspace(0,1,201)
	sns.displot(x='heterozygosity', hue='non-zero heterozygosity', data=het_data, bins=bins)
	plt.title('Heterozygosity Distribtuion (bin width: {:})'.format(bins[1]-bins[0]))
	plt.tight_layout()
	if save_plots:
		plt.savefig(os.path.join(save_dir, 'het_dist_bw{:}.png'.format(bins[1]-bins[0])))
	if show_plots:
		plt.show()

	sns.displot(x='heterozygosity', hue='non-zero heterozygosity', data=het_data, kind='ecdf')
	plt.title('Heterozygosity Distribtuion ECDF'.format(bins[1]-bins[0]))
	plt.tight_layout()
	if save_plots:
		plt.savefig(os.path.join(save_dir, 'het_dist_ecdf.png'))
	if show_plots:
		plt.show()

	# Plot the distibution of the entropy values
	ents = np.array([s['entropy'] for s in samples])
	ent_data = {'entropy': ents, 'non-zero entropy': (ents == 0)}
	sns.displot(x='entropy', hue='non-zero entropy', data=ent_data)
	plt.title('Entropy Distribtuion')
	plt.tight_layout()
	if save_plots:
		plt.savefig(os.path.join(save_dir, 'ent_dist.png'))
	if show_plots:
		plt.show()

	sns.displot(x='entropy', hue='non-zero entropy', data=ent_data, kind='ecdf')
	plt.title('Entropy Distribtuion ECDF')
	plt.tight_layout()
	if save_plots:
		plt.savefig(os.path.join(save_dir, 'ent_dist_ecdf.png'))
	if show_plots:
		plt.show()

	# Plot the distibution of the minor freq values
	mfrs = np.array([s['minor_freq'] for s in samples])
	mfr_data = {'minor_freq': mfrs, 'non-zero minor_freq': (mfrs == 0)}
	sns.displot(x='minor_freq', hue='non-zero minor_freq', data=mfr_data)
	plt.title('Minor Freq Distribtuion')
	plt.tight_layout()
	if save_plots:
		plt.savefig(os.path.join(save_dir, 'mfr_dist.png'))
	if show_plots:
		plt.show()

	sns.displot(x='minor_freq', hue='non-zero minor_freq', data=mfr_data, kind='ecdf')
	plt.title('Minor Freq Distribtuion ECDF')
	plt.tight_layout()
	if save_plots:
		plt.savefig(os.path.join(save_dir, 'mfr_dist_ecdf.png'))
	if show_plots:
		plt.show()

	# Plot the distibution of the minor counts
	mcs = np.array([s['minor_count'] for s in samples])
	mc_data = {'minor_count': mcs, 'non-zero minor_count': (mcs == 0)}
	sns.displot(x='minor_count', hue='non-zero minor_count', data=mc_data)
	plt.title('Minor Count Distribtuion')
	plt.tight_layout()
	if save_plots:
		plt.savefig(os.path.join(save_dir, 'mc_dist.png'))
	if show_plots:
		plt.show()

	sns.displot(x='minor_count', hue='non-zero minor_count', data=mc_data, kind='ecdf')
	plt.title('Minor Count Distribtuion ECDF')
	plt.tight_layout()
	if save_plots:
		plt.savefig(os.path.join(save_dir, 'mc_dist_ecdf.png'))
	if show_plots:
		plt.show()

	# Plot again with max copy number
	max_cn = 9
	hets = np.array([s['heterozygosity'] for s in samples if s['num_copies'] <= max_cn])
	het_data = {'heterozygosity': hets, 'non-zero heterozygosity': (hets == 0)}
	bins = np.linspace(0,1,201)
	sns.displot(x='heterozygosity', hue='non-zero heterozygosity', data=het_data, bins=bins)
	plt.title('Heterozygosity Distribtuion (bin width: {:}, max copy num: {:})'.format(
		bins[1]-bins[0], max_cn))
	plt.tight_layout()
	if save_plots:
		plt.savefig(os.path.join(save_dir, 'het_dist_bw{:}_maxcn{:}.png'.format(
			bins[1]-bins[0], max_cn)))
	if show_plots:
		plt.show()

	sns.displot(x='heterozygosity', hue='non-zero heterozygosity', data=het_data, kind='ecdf')
	plt.title('Heterozygosity Distribtuion ECDF (max copy num: {:})'.format(max_cn))
	plt.tight_layout()
	if save_plots:
		plt.savefig(os.path.join(save_dir, 'het_dist_ecdf_maxcn{:}.png'.format(max_cn)))
	if show_plots:
		plt.show()

	ents = np.array([s['entropy'] for s in samples if s['num_copies'] <= max_cn])
	ent_data = {'entropy': ents, 'non-zero entropy': (ents == 0)}
	sns.displot(x='entropy', hue='non-zero entropy', data=ent_data, kind='ecdf')
	plt.title('Entropy Distribtuion ECDF (max copy num: {:})'.format(max_cn))
	plt.tight_layout()
	if save_plots:
		plt.savefig(os.path.join(save_dir, 'ent_dist_ecdf_maxcn{:}.png'.format(max_cn)))
	if show_plots:
		plt.show()

	mfrs = np.array([s['minor_freq'] for s in samples if s['num_copies'] <= max_cn])
	mfr_data = {'minor_freq': mfrs, 'non-zero minor_freq': (mfrs == 0)}
	sns.displot(x='minor_freq', hue='non-zero minor_freq', data=mfr_data, kind='ecdf')
	plt.title('Minor Freq Distribtuion ECDF (max copy num: {:})'.format(max_cn))
	plt.tight_layout()
	if save_plots:
		plt.savefig(os.path.join(save_dir, 'mfr_dist_ecdf_maxcn{:}.png'.format(max_cn)))
	if show_plots:
		plt.show()

	mcs = np.array([s['minor_count'] for s in samples if s['num_copies'] <= max_cn])
	mc_data = {'minor_count': mcs, 'non-zero minor_count': (mcs == 0)}
	sns.displot(x='minor_count', hue='non-zero minor_count', data=mc_data, kind='ecdf')
	plt.title('Minor Count Distribtuion ECDF (max copy num: {:})'.format(max_cn))
	plt.tight_layout()
	if save_plots:
		plt.savefig(os.path.join(save_dir, 'mc_dist_ecdf_maxcn{:}.png'.format(max_cn)))
	if show_plots:
		plt.show()




	# Print counts by motif
	for m,c in zip(*np.unique(np.array([s['motif'] for s in samples]), 
								return_counts=True)):
		print(m, c)
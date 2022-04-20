import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':
	threshold_desc_str = 't0_005'

	# Load data
	data_dir = os.path.join('..', 'data', 'heterozygosity')
	data_fname = 'sample_data_dinucleotide_{}.json'.format(threshold_desc_str)

	with open(os.path.join(data_dir, data_fname)) as fp:    
		samples = json.load(fp)

	data_df = pd.DataFrame(samples)

	# Plot distributions
	counts_df = data_df.groupby(['num_copies',  'label']).count()
	sns.barplot(x='num_copies', y='str_seq', hue='label', data=counts_df.reset_index())
	plt.gcf().set_size_inches(10, 6)
	plt.ylabel('number of samples')
	plt.setp(plt.gca().get_xticklabels()[1::2], visible=False)
	plt.title('Heterozygosity by Copy Number')
	plt.tight_layout()
	plt.savefig(os.path.join(
		'length_and_stability', 
		'{}_heterozygosity_by_len.png'.format(threshold_desc_str)
	))
	plt.show()

	# STR len vs heterozygosity
	sns.lineplot(x='num_copies', y='heterozygosity', data=data_df, ci=99)
	plt.ylabel('heterozygosity')
	plt.title('Heterozygosity vs STR Length (99% confidence interval)')
	plt.tight_layout()
	plt.savefig(os.path.join(
		'length_and_stability', 
		'{}_heterozygosity_v_len.png'.format(threshold_desc_str)
	))
	plt.show()

	# STR len vs heterozygosity by motif
	data_df['motif type'] = data_df.motif.apply(lambda x: ''.join(sorted(x)))
	data_df.loc[data_df['motif type'].isin(['AC', 'GT']), 'motif type'] = 'AC/GT'
	data_df.loc[data_df['motif type'].isin(['AG', 'CT']), 'motif type'] = 'AG/CT'
	sns.lineplot(x='num_copies', y='heterozygosity', hue='motif type', data=data_df, ci=99)
	plt.ylabel('heterozygosity')
	plt.title('Heterozygosity vs STR Length (99% confidence interval)')
	plt.tight_layout()
	plt.savefig(os.path.join(
		'length_and_stability', 
		'{}_heterozygosity_v_len_by_type.png'.format(threshold_desc_str)
	))
	plt.show()

	# STR len vs heterozygosity heatmap
	sns.displot(x='num_copies', y='heterozygosity', data=data_df[data_df.label == 1], kind='kde')
	plt.ylabel('heterozygosity')
	plt.title('Heterozygosity vs STR Length Heatmap\n(heterozygous samples only)')
	plt.tight_layout()
	plt.savefig(os.path.join(
		'length_and_stability', 
		'{}_heterozygosity_v_len_heatmap.png'.format(threshold_desc_str)
	))
	plt.show()

	# Create table/plot of counts of each label with decreaing max copy number
	all_counts = []
	for m in range(15, 5, -1):
		counts_res = data_df[data_df.num_copies <= m].label.value_counts()
		all_counts.append({
			'max_copies': m,
			'stable_count': counts_res[0],
			'unstable_count': counts_res[1],
			'total_samples': counts_res.sum()
		})

	counts_df = pd.DataFrame(all_counts).reset_index(drop=True)
	counts_df.to_csv(os.path.join(
		'length_and_stability', 
		'{}_counts.csv'.format(threshold_desc_str)
	))

	melt_df = pd.melt(
		counts_df, 
		id_vars=['max_copies', 'total_samples'], 
		value_vars=['stable_count', 'unstable_count'],
		var_name='label',
		value_name='count'
	)
	sns.barplot(x='max_copies', y='count', hue='label', data=melt_df)
	plt.savefig(os.path.join(
		'length_and_stability', 
		'{}_label_counts_by_max_copy_num.png'.format(threshold_desc_str)
	))
	plt.show()
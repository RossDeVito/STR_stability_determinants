# """
# Compute correlations between STR length and heterozygosity for all STRs
# and plot length vs. heterozygosity for all.
# """

# import os

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.stats import pearsonr, spearmanr
# from tqdm import tqdm


# if __name__ == '__main__':
# 	data_path = os.path.join(
# 		'..', '..', 'data', 'heterozygosity', 'freqs_merged.tab'
# 	)

# 	# Load data
# 	panel_df = pd.read_csv(data_path, sep='\t')

# 	# Load STR region BED
# 	str_region_bed_path = os.path.join(
# 		'..', '..', 'data', 'HipSTR-references', 'GRCh38.hipstr_reference.bed.gz'
# 	)

# 	str_regions = pd.read_csv(
# 		str_region_bed_path, 
# 		sep='\t', 
# 		names=['chr', 'start', 'stop', 'motif_len', 
# 				'num_copies', 'str_name', 'motif'],
# 		low_memory=False, # because chr field is mixed type
# 	)
# 	str_regions['chr'] = str_regions.chr.map(lambda x: f'chr{x}')

# 	# Align by chrom and start/stop and use to label witgh motif and motif length
# 	chrom_dfs = dict()

# 	for chrom in panel_df.chrom.unique():
# 		chrom_dfs[chrom] = str_regions[str_regions.chr == chrom]

# 	labeled_samples = list()
# 	no_match = 0

# 	for chrom, chrom_df in tqdm(chrom_dfs.items()):
# 		chrom_panel_df = panel_df[panel_df.chrom == chrom]
# 		for _, row in tqdm(chrom_panel_df.iterrows(), total=len(chrom_panel_df), desc=chrom):
# 			# Find STR region
# 			# str_region = chrom_df[
# 			# 	(chrom_df.start <= row.pos) & (chrom_df.stop >= row.pos)
# 			# ]
# 			str_region = chrom_df[
# 				(chrom_df.start <= row.end) & (chrom_df.stop >= row.start)
# 			]
# 			if len(str_region) > 1:
# 				# select STR with greatest overlap
# 				overlap = str_region.apply(
# 					lambda x: min(x.stop, row.end) - max(x.start, row.start),
# 					axis=1
# 				)
# 				# str_region.iloc[(-overlap).argsort()]
# 				str_region = str_region.iloc[overlap.argmax()]

# 			# Label with motif and motif length
# 			if len(str_region) == 1:
# 				row['motif'] = str_region.iloc[0].motif
# 				row['motif_len'] = str_region.iloc[0].motif_len
# 				labeled_samples.append(row)
# 			else:
# 				no_match += 1
# 				# print("No STR region found for {}:{}:{}! ({} not found so far)".format(
# 				# 	chrom, row.start, row.end, no_match
# 				# ))

# 	print("No STR region found for {} samples".format(no_match))
	
# 	# Save
# 	labeled_samples_df = pd.DataFrame(labeled_samples)
# 	labeled_samples_df.to_csv(
# 		os.path.join('data', 'labeled_samples.csv'), index=False
# 	)